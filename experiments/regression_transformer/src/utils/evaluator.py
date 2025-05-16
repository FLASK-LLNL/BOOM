################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
"""
Taken from https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/trainer.py
"""
import collections
import os
from random import random
from time import time
from typing import Callable, List, Optional, Dict, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from selfies import encoder
from arguments import Arguments
from torch import nn
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from transformers import DataCollatorForPermutationLanguageModeling
from transformers.utils import logging
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)
from utils.trainer_utils import (
    DistributedTensorGatherer,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_pt_utils import ShardSampler

from tokenization.rt_collators import PropertyCollator
from utils.search import BeamSearch, GreedySearch, SamplingSearch
from utils.utils import find_safe_path

logger = logging.get_logger(__name__)

NON_MODEL_KEYS = ["real_property", "sample_weights"]

class Evaluator:
    def __init__(
        self,
        model: Module,
        args: Arguments,
        eval_params,
        data_collator: DataCollatorForPermutationLanguageModeling,
        eval_dataset: Dataset,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        prediction_loss_only: Optional[bool] = False,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None):

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.args_dict = vars(args)

        self.eval_batch_size = self.args.batch_size

        self.eval_dataset = eval_dataset
        self.eval_params = eval_params

        self.greedy_search = GreedySearch()
        self.sampling_search = SamplingSearch(
            temperature=eval_params.get("temperature", 1.0)
        )
        self.beam_search = BeamSearch(
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )

        # TODO: Resolve this in future version
        debugging = True
        if debugging:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
            rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
        self.world_size = world_size  #dist.get_world_size()
        self.rank = rank  #dist.get_rank()
        print(f"Rank, World Size: {self.rank}, {self.world_size}", flush=True)

        # Extract custom arguments
        self.verbose_evaluation = self.args_dict.get("verbose_evaluation",
                                                     True)
        logger.info(f"Verbose evaluation {self.verbose_evaluation}")


    def get_custom_dataloader(self, collator, bs: int = -1) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`nlp.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed.
        """
        # FIXME: Commenting this out to debug but we should probably update 
        #   BaseCollator to be subclass of DataCollatorForPermutationLanguageModeling
        #if not isinstance(collator, DataCollatorForPermutationLanguageModeling):
        #    raise TypeError(f"Needs PLM collator not {type(collator)}")
        return DataLoader(
            self.eval_dataset,
            sampler=self._get_eval_sampler(self.eval_dataset),
            batch_size=self.eval_batch_size if bs == -1 else bs,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def multi_property_prediction(self, collator, save_path=None, rmse_factor: int = 1):

        eval_dataloader = self.get_custom_dataloader(collator)
        # Forward pass
        logits, label_ids, metrics, input_ids = self.prediction_loop(
            dataloader=eval_dataloader,
            description=f"Predicting {collator.property_tokens}",
            prediction_loss_only=False,
            return_inputs=True,
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        keep_pos = lambda x: not np.logical_or((x == -100), (x == 0)).all()
        pos_to_keep = [i for i in range(logits.shape[1]) if keep_pos(label_ids[:, i])]
        relevant_logits = torch.Tensor(logits[:, pos_to_keep, :])

        # Compute performance (only on the logits where predictions are relevavnt)
        print("Obtained logits, now performing beam search...")
        greedy_preds = self.greedy_search(relevant_logits).unsqueeze(0)
        sampling_preds = self.sampling_search(relevant_logits).unsqueeze(0)
        beam_preds, scores = self.beam_search(relevant_logits)
        beam_preds = beam_preds.permute(2, 0, 1)

        # Reassign full prediction matrices (0 means mask)
        all_preds = torch.zeros(2 + beam_preds.shape[0], *logits.shape[:2]).long()
        all_preds[0, :, pos_to_keep] = greedy_preds
        all_preds[1, :, pos_to_keep] = sampling_preds
        # In case beam width > 0:
        for k in range(beam_preds.shape[0]):
            all_preds[2 + k, :, pos_to_keep] = beam_preds[k, :, :]

        # Define rmse function
        rmse = lambda x, y: np.sqrt(sum((np.array(x) - np.array(y)) ** 2) / len(x))

        num_props = len(collator.property_tokens)
        num_decoders = 2 + beam_preds.shape[0]
        num_samples = len(relevant_logits)

        property_labels = torch.zeros(num_props, num_samples)
        property_predictions = torch.zeros(num_props, num_decoders, num_samples)
        pearsons = np.zeros((num_props, num_decoders))
        rmses = np.zeros((num_props, num_decoders))
        for idx, predictions in enumerate(all_preds):

            for sidx, (x, y, yhat) in enumerate(zip(input_ids, label_ids, predictions)):

                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")
                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                yhat_tokens = self.tokenizer.decode(
                    yhat, clean_up_tokenization_spaces=False
                ).split(" ")

                # Get label (float)
                label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                # Get prediction (float)
                gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)

                _, target_prop = self.tokenizer.aggregate_tokens(label, label_mode=True)
                _, gen_prop = self.tokenizer.aggregate_tokens(gen, label_mode=False)

                assert target_prop.keys() == gen_prop.keys()
                # Save predictions for all properties
                for i, p in enumerate(target_prop.keys()):
                    if idx == 0:
                        property_labels[i, sidx] = target_prop[p] / rmse_factor
                    property_predictions[i, idx, sidx] = gen_prop[p] / rmse_factor

            for i, prop in enumerate(collator.property_tokens):
                p = pearsonr(property_labels[i, :], property_predictions[i, idx, :])
                r = rmse(property_labels[i, :], property_predictions[i, idx, :])
                if idx == 0:
                    print(f"Pearson for {prop} is {p[0]:.3f}")
                    print(f"RMSE for {prop} is {r:.3f}")
                pearsons[i, idx] = p[0]
                rmses[i, idx] = r

        if save_path is not None:

            bw = self.beam_search.beam_width
            beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
            search_cols = ["Label", "Greedy", "Sampling"] + beam_cols
            for i, prop in enumerate(collator.property_tokens):
                pd.DataFrame(
                    np.concatenate(
                        [
                            property_labels[i, :].unsqueeze(1).numpy(),
                            property_predictions[i, :, :].T.numpy(),
                        ],
                        axis=1,
                    ),
                    columns=search_cols,
                ).to_csv(f"{save_path}_predict_{prop[1:-1]}.csv")

        return pearsons, rmses

    def property_prediction(self, collator, save_path=None, rmse_factor: int = 1):
        """
        Predict the property
        """

        eval_dataloader = self.get_custom_dataloader(collator)

        # Hack for XLNet tokenizer
        self.tokenizer.real_decoder = self.tokenizer.decode
        #self.tokenizer.decode = self.tokenizer.decode_internal #Original version-EA Edited
        self.tokenizer.decode=self.tokenizer.decode

        for prop in collator.property_tokens:

            # Forward pass
            logits, label_ids, metrics, input_ids = self.prediction_loop(
                dataloader=eval_dataloader,
                description=f"Predicting {prop}",
                prediction_loss_only=False,
                return_inputs=True,
                pad_idx=self.tokenizer.vocab["[PAD]"],
            )
            _prop = prop[1:-1]

            """
            NOTE: Saving time by only running search on affected positions
            Sequence positions where the label was not -100 (MASK) or 0 (PAD) at least
            once. Those positions are used for the searches. This heavily bases on the
            assumption that the positions are more or less *stable* across the samples
            (good for property prediction but for CD, it's less efficient).
            """
            keep_pos = lambda x: not np.logical_or((x == -100), (x == 0)).all()
            pos_to_keep = [
                i for i in range(logits.shape[1]) if keep_pos(label_ids[:, i])
            ]
            relevant_logits = torch.Tensor(logits[:, pos_to_keep, :])

            # Compute performance (only on the logits where predictions are relevavnt)
            print("Obtained logits, now performing beam search...")
            greedy_preds = self.greedy_search(relevant_logits).unsqueeze(0)
            sampling_preds = self.sampling_search(relevant_logits).unsqueeze(0)
            # beam_preds, scores = self.beam_search(relevant_logits)
            # beam_preds = beam_preds.permute(2, 0, 1)

            # Reassign full prediction matrices (0 means mask)
            all_preds = torch.zeros(2, *logits.shape[:2]).long()
            all_preds[0, :, pos_to_keep] = greedy_preds
            all_preds[1, :, pos_to_keep] = sampling_preds
            # In case beam width > 0:
            # for k in range(beam_preds.shape[0]):
            #     all_preds[2 + k, :, pos_to_keep] = beam_preds[k, :, :]

            # Define rmse function
            rmse = lambda x, y: np.sqrt(sum((np.array(x) - np.array(y)) ** 2) / len(x))

            property_labels = torch.zeros(len(relevant_logits))
            property_predictions = torch.zeros(len(relevant_logits), len(all_preds))
            pearsons, rmses, spearmans = (
                np.zeros((len(all_preds))),
                np.zeros((len(all_preds))),
                np.zeros((len(all_preds))),
            )
            for idx, predictions in enumerate(all_preds):

                for sidx, (x, y, yhat) in enumerate(
                    zip(input_ids, label_ids, predictions)
                ):

                    x_tokens = self.tokenizer.decode(
                        x, clean_up_tokenization_spaces=False
                    ).split(" ")
                    y_tokens = self.tokenizer.decode(
                        y, clean_up_tokenization_spaces=False
                    ).split(" ")
                    yhat_tokens = self.tokenizer.decode(
                        yhat, clean_up_tokenization_spaces=False
                    ).split(" ")
                    # print('yhat tokens', ''.join(yhat_tokens))
                    # print('y_tokens', ''.join(y_tokens))
                    # print('y', y)

                    # Get label (float)
                    label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                    # Get prediction (float)
                    gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)

                    _, target_prop = self.tokenizer.aggregate_tokens(
                        label, label_mode=True
                    )
                    # print('target_prop is', target_prop)
                    _, gen_prop = self.tokenizer.aggregate_tokens(gen, label_mode=False)
                    # print('gen_prop is', gen_prop)

                    property_predictions[sidx, idx] = (
                        gen_prop.get(_prop, -1) / rmse_factor
                    )
                    if idx == 0:
                        property_labels[sidx] = target_prop[_prop] / rmse_factor

                p = pearsonr(property_labels, property_predictions[:, idx])
                r = rmse(property_labels, property_predictions[:, idx])
                s = spearmanr(property_labels, property_predictions[:, idx])
                if idx == 0:
                    print(f"Pearson is {p[0]}")
                    print(f"RMSE is {r}")
                    print(f"Spearman is {s[0]}")
                else:
                    print("SAMPLING Preds")
                    print(f"Pearson is {p[0]}")
                    print(f"RMSE is {r}")
                    print(f"Spearman is {s[0]}")
                pearsons[idx] = p[0]
                rmses[idx] = r
                spearmans[idx] = s[0]
            if save_path is not None:
                bw = self.beam_search.beam_width
                bw = 0
                beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
                search_cols = ["Label", "Greedy", "Sampling"] + beam_cols
                os.makedirs(os.path.join(save_path, "checkpoint-rmse-min-2400"),exist_ok=True) #EA Edited
                pd.DataFrame(
                    np.concatenate(
                        [
                            property_labels.unsqueeze(1).numpy(),
                            property_predictions.numpy(),
                        ],
                        axis=1,
                    ),
                    columns=search_cols,
                ).to_csv(
                    os.path.join(save_path, "checkpoint-rmse-min-2400", "test.csv")
                )

        self.tokenizer.decode = self.tokenizer.real_decoder

        return pearsons, rmses, spearmans

    def conditional_generation(
        self,
        collator,
        save_path: str = None,
        passed_eval_fn: Callable = None,
        property_collator=None,
        denormalize_params: Optional[List[float]] = None,
    ):
        """
        Function to evaluate conditional generation

        Args:
            collator (): PyTorch collator object
            save_path (str): Path where results are saved, defaults to None (no saving).
            passed_eval_fn (Callable): Function used to evaluate whether the generated molecules
                adhere to the property of interest. Defaults to None, in which case the model
                is used *itself* to evaluate the molecules (NOTE: This can be a biased estimator).
                NOTE: Function should be callable with a SMILES string.
            property_collator (): PyTorch collator object. Defaults to None. Only needed if passed_eval_fn
                is None
            denormalize_params: The min and max values of the property to denormalize
                the results.
        """

        if passed_eval_fn is None and property_collator is None:
            raise ValueError(
                "If model should be used for evaluation, property collator is required"
            )

        eval_dataloader = self.get_custom_dataloader(collator, bs=2)
        prop = collator.property_token[1:-1]

        if passed_eval_fn is not None:
            eval_fn = passed_eval_fn
        else:
            eval_fn = self.get_seq_eval_fn(
                collator=property_collator, prefix=f"<{prop}>0.123|"
            )
        if denormalize_params:
            denormalize = (
                lambda x: x * (denormalize_params[1] - denormalize_params[0])
                + denormalize_params[0]
            )
        else:
            denormalize = lambda x: x

        seq_to_prop = {}

        # Forward pass
        logits, label_ids, metrics, input_ids, returned = self.prediction_loop(
            dataloader=eval_dataloader,
            description=f"Conditional generation {prop}",
            prediction_loss_only=False,
            return_inputs=True,
            pop_and_return_keys=["real_property", "sample_weights"],
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        logits = torch.Tensor(logits).cpu()
        input_ids = torch.Tensor(input_ids)
        # Arbitrary rounding set here
        real_prop = [round(denormalize(x), 4) for x in returned["real_property"]]

        # Naive search (using all tokens)
        t = time()
        greedy_preds = self.greedy_search(logits).unsqueeze(0)
        logger.error(f"Greedy search took {time() - t}")
        t = time()
        sampling_preds = self.sampling_search(logits).unsqueeze(0)
        logger.error(f"Sampling search took {time() - t}")
        # Just needed for reference
        bw = self.beam_search.beam_width
        beam_preds = torch.cat([greedy_preds] * bw, dim=0)

        # Restrict beam search to affected logits
        t = time()
        for sample_idx in tqdm(range(logits.shape[0]), desc="Beam search"):
            keep_pos = torch.nonzero(
                input_ids[sample_idx, :] == self.tokenizer.vocab["[MASK]"]
            ).squeeze(1)
            relevant_logits = logits[sample_idx, keep_pos, :].unsqueeze(0)
            if len(relevant_logits.shape) == 2:
                relevant_logits = relevant_logits.unsqueeze(0)
            beams, scores = self.beam_search(relevant_logits)
            beam_preds[:, sample_idx, keep_pos] = (
                beams.squeeze(dim=0).permute(1, 0).long()
            )
        print(f"Beam search took {time() - t}")

        all_preds = torch.zeros(2 + beam_preds.shape[0], *logits.shape[:2]).long()
        all_preds[0, :, :] = greedy_preds
        all_preds[1, :, :] = sampling_preds

        # In case beam width > 0:
        if bw > 0:
            for k in range(beam_preds.shape[0]):
                all_preds[2 + k, :, :] = beam_preds[k, :, :]

        array_len = logits.shape[0] * all_preds.shape[0]
        property_primers = torch.zeros(array_len)
        property_generations = torch.zeros(array_len)
        original_seqs = np.empty(array_len, dtype="object")
        generated_seqs = np.empty(array_len, dtype="object")
        prop_dicts = []

        for idx, predictions in tqdm(enumerate(all_preds), desc="Evaluating search"):
            for sidx, (x, y, yhat) in tqdm(
                enumerate(zip(input_ids, label_ids, predictions))
            ):

                cidx = idx * len(predictions) + sidx
                assert len(x) == len(y), "Input and label lengths do not match"
                assert len(x) == len(yhat), "Input and predictions do not match"
                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")
                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                yhat_tokens = self.tokenizer.decode(
                    yhat, clean_up_tokenization_spaces=False
                ).split(" ")
                assert len(x_tokens) == len(
                    y_tokens
                ), f"I/O lengths must match  {len(x_tokens)} and {len(y_tokens)}"
                # Get property primer (float)
                label = self.tokenizer.get_sample_label(y_tokens, x_tokens)
                # Get prediction (float)
                gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)
                orgseq, target_prop = self.tokenizer.aggregate_tokens(
                    label, label_mode=True
                )
                original_seqs[cidx] = orgseq.split("|")[-1]

                genseq, _ = self.tokenizer.aggregate_tokens(gen, label_mode=False)
                generated_seqs[cidx] = genseq.split("|")[-1]
                gen_seq = self.tokenizer.to_readable(generated_seqs[cidx])

                # Checking whether molecule was already predicted
                if gen_seq in seq_to_prop.keys():
                    prop_dict = seq_to_prop[gen_seq]
                    value = prop_dict[prop.lower()]
                else:
                    value, prop_dict = eval_fn(gen_seq)
                    value = denormalize(value)
                    prop_dict[prop.lower()] = round(value, 4)
                prop_dicts.append(prop_dict)

                # except Exception:
                #     value = -1
                #     prop_dicts.append({})
                seq_to_prop[gen_seq] = prop_dict

                property_generations[cidx] = value
                property_primers[cidx] = denormalize(target_prop[prop])

        pg = property_generations[property_generations != -1]
        print(f"Ratio of invalid sequences: {round(1 - (len(pg)/array_len),2)}")
        pp = property_primers[property_generations != -1]
        p = pearsonr(pp, pg)
        s = spearmanr(pp, pg)

        print(f"Global Pearson is {round(p[0], 3)} ({p[1]})")
        print(f"Global Spearman is {round(s[0], 3)} ({s[1]})")

        if save_path is not None:
            beam_cols = ["Beam"] if bw == 1 else [f"Beam{i}" for i in range(bw)]
            search_cols = ["Greedy", "Sampling"] + beam_cols
            prop = prop.capitalize()
            save_path = find_safe_path(save_path)

            df = pd.DataFrame(
                {
                    "SeedSequence": original_seqs,
                    f"Seed{prop}": np.tile(
                        np.repeat(real_prop, collator.num_primed), len(search_cols)
                    ),
                    f"Primer{prop}": property_primers,
                    "GenSequence": generated_seqs,
                    f"Gen{prop}": property_generations.tolist(),
                    "Search": np.repeat(
                        search_cols, collator.num_primed * len(real_prop)
                    ),
                }
            )
            remaining_props = pd.DataFrame(prop_dicts)
            replacer = dict(
                zip(
                    remaining_props.columns,
                    [f"Gen{k.capitalize()}" for k in remaining_props.columns],
                )
            )
            remaining_props = remaining_props.rename(columns=replacer)
            remaining_props = remaining_props.drop(columns=[f"Gen{prop}"])
            df = pd.concat([df, remaining_props], axis=1)
            df["sort_helper"] = df.Search.apply(
                lambda x: {"Greedy": 0, "Sampling": 2}.get(x, 1)
            )
            df = df.sort_values(
                by=["SeedSequence", "Search", f"Primer{prop}"],
                ascending=[True, False, True],
            ).drop_duplicates(subset=["GenSequence", "SeedSequence"])
            df.to_csv(os.path.join(save_path))
            print(f"Data frame has {len(df)} samples, saved in {save_path}")
            spearmans = [
                spearmanr(
                    df[(df.SeedSequence == x) & (df.Search == "Sampling")][
                        f"Primer{prop}"
                    ],
                    df[(df.SeedSequence == x) & (df.Search == "Sampling")][
                        f"Gen{prop}"
                    ],
                )[0]
                for x in df.SeedSequence.unique()
            ]
            print(
                f"Average per sample Spearman for sampling search: {np.nanmean(spearmans)}"
            )

    def get_seq_eval_fn(self, collator: PropertyCollator, prefix: str) -> Callable:
        """
        Returns a function that can be called with a sequence and returns the predicted
        property by the model. The property token is being set by the collator.

        Args:
            collator (PropertyCollator): A property collator

        Returns:
            Callable: Function to be called with a text string
        """
        from utils.property_predictors import PREDICT_FACTORY

        property_token = collator.property_tokens[0]
        # Check whether shortcut is possible
        if property_token in PREDICT_FACTORY.keys():
            return PREDICT_FACTORY[property_token]

        def eval_fn(seq):
            if not isinstance(seq, str):
                return -1, {property_token: -1}
            if self.tokenizer.language == "SELFIES":
                seq = encoder(seq)
            input_str = f"{prefix}{seq}"
            sample_ids = [torch.tensor(self.tokenizer(input_str)["input_ids"])]
            prepared_inputs = collator(sample_ids)
            _, logits, _ = self.prediction_step(
                model=self.model, inputs=prepared_inputs, prediction_loss_only=False
            )
            greedy_preds = self.greedy_search(logits).squeeze(0)
            x_tokens = self.tokenizer.decode(
                prepared_inputs["input_ids"].squeeze()
            ).split(" ")
            yhat_tokens = self.tokenizer.decode(greedy_preds).split(" ")
            gen = self.tokenizer.get_sample_prediction(yhat_tokens, x_tokens)
            _, prop_dict = self.tokenizer.aggregate_tokens(gen, label_mode=False)
            value = prop_dict[property_token[1:-1]]
            return value, prop_dict

        return eval_fn

    def cg_evaluate(self, dataloader, k: int = 10):
        """
        Function to evaluate funnyness dataset joke generation
        """

        # Forward pass
        logits, label_ids, metrics, input_ids = self.prediction_loop(
            dataloader=dataloader,
            description="Conditional generation",
            prediction_loss_only=False,
            return_inputs=True,
            pad_idx=self.tokenizer.vocab["[PAD]"],
        )
        num_samples = logits.shape[0]
        logits = torch.Tensor(logits).cpu()
        input_ids = torch.Tensor(input_ids)

        topk_values, topk_indices = torch.topk(logits, k=k, dim=2)
        all_preds = topk_indices.permute(2, 0, 1)

        # self.tokenizer.decode = self.tokenizer.decode_internal
        accuracies = np.zeros((num_samples, k))
        sentences = {"real": []}
        for _k in range(k):
            sentences[f"top_{_k}"] = []

        for idx, predictions in tqdm(enumerate(all_preds), desc="Evaluating search"):
            for sidx, (x, y, yhat) in tqdm(
                enumerate(zip(input_ids, label_ids, predictions))
            ):
                cidx = idx * len(predictions) + sidx
                x[x == -100] = 6
                y[y == -100] = 6
                yhat[yhat == -100] = 6
                assert len(x) == len(y), "Input and label lengths do not match"
                assert len(x) == len(yhat), "Input and predictions do not match"
                x_tokens = self.tokenizer.decode(
                    x, clean_up_tokenization_spaces=False
                ).split(" ")

                y_tokens = self.tokenizer.decode(
                    y, clean_up_tokenization_spaces=False
                ).split(" ")
                _x_tokens = []
                for x in x_tokens:
                    if "{" in x and "}" in x and "<mask>" in x:
                        _x_tokens.extend(["{"] + ["<mask>"] * x.count("<mask>") + ["}"])
                    elif "<sep>" in x:
                        _x_tokens.append(x.split("<sep>")[0])
                    else:
                        _x_tokens.append(x)

                x_tokens = []
                for x in _x_tokens:
                    if "[PAD]" in x:
                        break
                    x_tokens.append(x)
                _y_tokens = []
                for y in y_tokens:
                    if y.startswith("<mask>"):
                        _y_tokens.extend(y.split("<mask>")[1:])
                    else:
                        _y_tokens.extend(y.split("<mask>"))
                y_tokens = _y_tokens

                # Get sample prediction
                s, e = x_tokens.index(
                    self.tokenizer.expression_separator
                ), x_tokens.index(self.tokenizer.expression_end)
                predicted = self.tokenizer.decode(
                    yhat[s + 1 : e], clean_up_tokenization_spaces=False
                )
                predicted = predicted.split(" ") if predicted != "" else [""]
                predicted = "".join(predicted)

                label = "".join(y_tokens[s + 1 : e])
                if predicted == label:
                    accuracies[sidx, idx] = 1

                # Real sentence
                real_sentence = x_tokens[e + 1 :]
                joke_idxs = list(
                    range(real_sentence.index("START") + 1, real_sentence.index("END"))
                )
                predicted_sentence = []
                filled = False
                for i, x in enumerate(real_sentence):
                    if (i not in joke_idxs) or filled:
                        predicted_sentence.append(x)
                    else:
                        # Fill joke
                        predicted_sentence.append(predicted)
                        filled = True

                # Change real sentence s.t. the joke is replaced with ground truth
                for i, j in enumerate(joke_idxs):
                    real_sentence[j] = y_tokens[s + 1 + i]

                real_sentence = " ".join(
                    [x for x in real_sentence if x not in ["START", "END"]]
                )

                predicted_sentence = " ".join(
                    [x for x in predicted_sentence if x not in ["START", "END"]]
                )
                if idx == 0:
                    sentences["real"].append(real_sentence)
                sentences[f"top_{idx}"].append(predicted_sentence)

        topk_accuracies = compute_topk(accuracies)
        # self.tokenizer.decode = self.tokenizer.real_decode
        return topk_accuracies, sentences

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if self.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.world_size,
                process_index=self.args.process_index,
            )

    def evaluate(
        self,
        #eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        eval_dataloader: DataLoader,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        ## handle multipe eval datasets
        #eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        #if isinstance(eval_dataset, dict):
        #    metrics = {}
        #    for eval_dataset_name, _eval_dataset in eval_dataset.items():
        #        dataset_metrics = self.evaluate(
        #            eval_dataset=_eval_dataset,
        #            ignore_keys=ignore_keys,
        #            metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
        #        )
        #        metrics.update(dataset_metrics)
        #    return metrics

        #eval_dataloader = self.get_eval_dataloader(eval_dataset)
        #if self.is_fsdp_xla_v2_enabled:
        #    eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time()

        eval_loop = self.prediction_loop  #if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=
            True,  # if self.compute_metrics is None else None,
            #ignore_keys=ignore_keys,
            #metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.eval_batch_size * self.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                #num_samples=output.num_samples,
                #num_steps=math.ceil(output.num_samples / total_batch_size),
            ))

        self.log(output.metrics)

        # TODO: remove these lines after next commit
        #self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return output.metrics


    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        return_inputs: bool = False,
        pop_and_return_keys: Optional[List[str]] = None,
        pad_idx: int = 100,
    ) -> PredictionOutput:
        """
        NOTE: Overwritten because
            - fixing tensor stacking https://github.com/huggingface/transformers/issues/7584
            - enable eval_accumulation_steps (introduced only in 3.4.0)
            - to return the inputs

        pop_and_return_keys (Optional[List[str]]): List of keys for the `inputs` dict
            produced by the collator. If passed, each item of list is popped from dict
            *before* calling the model and returned. Defaults to None.

        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only)
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        prediction_loss_only = (prediction_loss_only
                                if prediction_loss_only is not None else
                                self.args.prediction_loss_only)

        model = self.model
        # multi-gpu eval
        if self.world_size > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        if self.args.dataloader_drop_last:
            num_examples -= (num_examples) % batch_size
        num_primed = (dataloader.collate_fn.num_primed if hasattr(
            dataloader.collate_fn, "num_primed") else 1)

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

        num_examples = num_examples * num_primed
        eval_losses_gatherer = DistributedTensorGatherer(
            self.world_size, num_examples, make_multiple_of=batch_size)
        preds_gatherer = DistributedTensorGatherer(self.world_size,
                                                   num_examples)
        labels_gatherer = DistributedTensorGatherer(self.world_size,
                                                    num_examples)
        inputs_gatherer = DistributedTensorGatherer(self.world_size,
                                                    num_examples)

        if pop_and_return_keys:
            return_collator_data = {k: list() for k in NON_MODEL_KEYS}

        # eval_losses: List[float] = []
        # preds: torch.Tensor = None
        # label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        # Set up alternating iterator (if applicable)
        #alt_loader = self.alt_eval_loader if self.alt_training else dataloader


        # NOTE: The below is a mixture of transformers 3.1.0 and 3.4.0. 3.4.0 introduced
        # the CallbackHandler class which effectively requires rewriting large parts
        # of the package.
        disable_tqdm = not self.is_local_process_zero(
        ) or self.args.disable_tqdm
        samples_count = 0

        epoch_pbar = tqdm(dataloader, desc=description, disable=disable_tqdm)
        #for step, (inputs, a_inputs) in enumerate(zip(dataloader, alt_loader)):
        for step, inputs in enumerate(dataloader):

            epoch_pbar.update(1)
            # To optionally take out keys from the collator dict.
            if pop_and_return_keys:
                for k in pop_and_return_keys:
                    return_collator_data[k].extend(
                        inputs.get(k, torch.Tensor()).tolist())
                    inputs.pop(k, None)

            # TODO: Remove all traces of alt_training/alt_loader
            #if self.alt_training:
            #    self.cg_mode = self.get_cg_mode(step)
            #    if (self.get_cg_mode(step) > self.get_cg_mode(step - 1)
            #            and step % 100 == 0):
            #        logger.debug("Switching to CG task")
            #    if self.cg_mode:
            #        inputs = a_inputs

            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            samples_count += batch_size
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0))
                # eval_losses.append(loss * batch_size)
            if logits is not None:
                preds_host = (logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=pad_idx))
            if labels is not None:
                labels_host = (labels
                               if labels_host is None else nested_concat(
                                   labels_host, labels, padding_index=pad_idx))
            if inputs is not None:
                inputs_host = (inputs["input_ids"] if inputs_host is None else
                               nested_concat(inputs_host,
                                             inputs["input_ids"],
                                             padding_index=pad_idx))

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (self.args.eval_accumulation_steps > 0  #is not None
                    and (step + 1) % self.args.eval_accumulation_steps == 0):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses"))
                preds_gatherer.add_arrays(
                    self._gather_and_numpify(preds_host, "eval_preds"))
                labels_gatherer.add_arrays(
                    self._gather_and_numpify(labels_host, "eval_label_ids"))
                inputs_gatherer.add_arrays(
                    self._gather_and_numpify(inputs_host, "eval_input_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, inputs_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses"))
        preds_gatherer.add_arrays(
            self._gather_and_numpify(preds_host, "eval_preds"))
        labels_gatherer.add_arrays(
            self._gather_and_numpify(labels_host, "eval_label_ids"))
        inputs_gatherer.add_arrays(
            self._gather_and_numpify(inputs_host, "eval_input_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize()
        label_ids = labels_gatherer.finalize()
        input_ids = inputs_gatherer.finalize()

        if (self.compute_metrics is not None and preds is not None
                and label_ids is not None):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss.mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        if not return_inputs:
            return PredictionOutput(predictions=preds,
                                    label_ids=label_ids,
                                    metrics=metrics)
        elif return_inputs and not pop_and_return_keys:
            return preds, label_ids, metrics, input_ids
        elif return_inputs and pop_and_return_keys:
            return preds, label_ids, metrics, input_ids, return_collator_data


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
    ) -> Tuple[Optional[float], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        NOTE: Overwritten here to enable custom embeddings + for moinitoring purposes.

        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """

        has_labels = any(
            inputs.get(k) is not None
            for k in ["labels", "lm_labels", "masked_lm_labels"])

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # NOTE: Overwritten with custom embeddings
            outputs = self.feed_model(model, inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().detach()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else
                                     self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        logits = logits.detach()

        # NOTE: Overwritten for moinitoring purposes (will print occassionally)
        if self.verbose_evaluation and random() < 0.00001:

            try:
                # TODO: Only fill the masked tokens
                prediction = (self.search(logits[1, :, :].unsqueeze(
                    0)).detach().cpu().squeeze().tolist())
                gt_seq, gt_dict = self.tokenizer.aggregate_tokens(
                    self.tokenizer.get_sample_label(
                        self.tokenizer.convert_ids_to_tokens(labels[0]),
                        self.tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0]),
                    ),
                    label_mode=True,
                )

                p_seq, p_dict = self.tokenizer.aggregate_tokens(
                    self.tokenizer.convert_ids_to_tokens(prediction),
                    label_mode=False)

                logger.info(
                    f"\nPredicted: {p_seq} \t, {p_dict.get('qed', -1)}")
                logger.info(
                    f"Ground truth {gt_seq} \t {gt_dict.get('qed', -1)}")
            except Exception:
                logger.info("Error occurred in converting logits to sequence.")

        return loss, logits, labels


    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            #if isinstance(dataset, IterableDatasetShard):
            #    return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError
                ):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_batch_size

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_rank == 0


    def feed_model(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Forward pass of `inputs` through `model`. This function handles the numerical
        encodings if applicable.

        Args:
            model (nn.Module): The model to consume data.
            inputs (Dict[str, Union[torch.Tensor, Any]]): A dict that can be understood
                by model.__call__. Keys should include `input_ids`, `perm_mask`,
                `labels` and `target_mapping`.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Output from model
        """
        model_inputs = inputs  # shallow copy

        #if self.alt_training and self.cg_mode:
        #    model_inputs = inputs.copy()
        [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
        outputs = model(**model_inputs)

        ## If we enter here, we are in a training step where we are doing cond. gen.
        #if self.alt_training and self.cg_mode:

        #    # Apply conditional generation loss (BCE loss on affected tokens) with
        #    # custom sample weights to reflect
        #    #   1) distance of sampled condition to real prop (vanilla CG collator)
        #    #   2) "negative learning" (Bimodal generator)
        #    logits = outputs[1].permute(0, 2, 1)
        #    sample_loss = self.cg_bce_loss_fn(logits,
        #                                      inputs["labels"]).mean(axis=-1)

        #    loss = (inputs["sample_weights"] * sample_loss).mean()

        #    if self.cc_loss:
        #        # Apply cycle-consistency loss
        #        cc_loss = self.get_cc_loss(model, inputs, outputs)
        #        loss += cc_loss * self.cc_loss_weight

        #    # Overwrite PLM loss with CC loss
        #    outputs = (loss, *outputs[1:])

        return outputs


    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        elif self.args.local_rank != -1:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)



def compute_topk(predictions: np.array) -> List[float]:
    """
    Computes the topk accuracy of a boolean np array

    Args:
        predictions: boolean np.array of shape batch_size x k with correctness of each
            prediction

    Returns:
        List of floats denoting the top-k accuracies
    """

    topk = [np.mean(predictions[:, 0])]
    for k in range(1, predictions.shape[1]):
        topk.append(topk[-1] + np.mean(predictions[:, k]))
    return topk
