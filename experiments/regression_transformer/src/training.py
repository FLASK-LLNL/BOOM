################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from arguments import Arguments
from torch.nn import Module

import collections
import gc
import json
import os
import shutil
import warnings
from random import random
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import DataCollatorForPermutationLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from torch.optim import AdamW
from scipy.stats import pearsonr, spearmanr
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm, trange

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    set_seed,
    speed_metrics,
)

from tokenization.rt_collators import BaseCollator, TRAIN_COLLATORS
from tokenization.rt_tokenizers import ExpressionBertTokenizer

# Required from original version of regression transformer
from utils.search import SEARCH_FACTORY
from utils.trainer_utils import (
    DistributedTensorGatherer,
    distributed_concat,
    get_trainer_dict,
    nested_concat,
    nested_numpify,
)
from utils.optimizer import get_scheduler

import logging

logger = logging.getLogger(__name__)

NON_MODEL_KEYS = ["real_property", "sample_weights"]

# TODO: Missing functions from Evaluator
#get_custom_dataloader
#conditional_generation
#get_seq_eval_fn
#cg_evaluate

class Trainer:

    def __init__(
        self,
        model: Module,
        args: Arguments,
        property_tokens: List[str],
        eval_loader_cg: DataLoader,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        train_config: Optional[dict] = {},
        prediction_loss_only: Optional[bool] = False,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None)):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.property_tokens = property_tokens
        self.compute_metrics = compute_metrics
        # Setup tensorboard writer (if used)
        if self.args.no_tboard:
            self.tb_writer = None
        else:
            tboard_log_folder = (f"tensorboard-logs")
            tboard_log_dir = os.path.join(self.args.output_dir,
                tboard_log_folder)
            self.tb_writer = SummaryWriter(tboard_log_dir)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
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

        self.args_dict = vars(args)

        self.train_batch_size = self.args.batch_size
        self.eval_batch_size = self.args.batch_size

        # Setup Sharded DDP training (FIXME: Currently will always be False, requires external package)
        self.sharded_dpp = False
        #if args.sharded_ddp:
        #    if args.local_rank == -1:
        #        raise ValueError("Using sharded DDP only works in distributed training.")
        #    elif not is_fairscale_available():
        #        raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
        #    elif args.deepspeed:
        #        raise ValueError("can't use --sharded_ddp together with --deepspeed.")
        #    else:
        #        self.sharded_dpp = True

        # Instantiate optimizer and scheduler (if they are None they will
        # be instantiated to default in `create_optimizer_and_scheduler`)
        self.optimizer, self.lr_scheduler = optimizers

        # Extract custom arguments
        self.verbose_evaluation = self.args_dict.get("verbose_evaluation",
                                                     True)
        logger.info(f"Verbose evaluation {self.verbose_evaluation}")

        # Restore the logged parameters (training)
        self.logs = self.args_dict.get("training_logs", [])
        self.eval_logs = []

        # Will save RMSE and Pearson of every epoch
        try:
            tokens = self.property_tokens
        except AttributeError:
            tokens = [None]

        self.perfs = np.tile(
            np.expand_dims(np.vstack([10e5 * np.arange(3)] * len(tokens)), -1),
            10000,
        ).astype(float)
        self.cidx = 0

        if self.logs != []:
            self.min_loss = pd.DataFrame(self.logs)["loss"].min()
            if self.args_dict.get("reset_training_loss", False):
                self.min_loss = 10e5
            logger.info(f"Current minimal loss {self.min_loss}")
        else:
            self.min_loss = 10e5

        self.search = SEARCH_FACTORY[self.args_dict.get(
            "eval_search",
            "greedy")](self.args_dict.get("eval_search_args", {}))

        self.alt_training = False
        if 'alternate_tasks' in train_config:
            self.alt_training = train_config['alternate_tasks']

        self.save_attention = self.args_dict.get("save_attention", False)

        # Whether we train regular PLM or alternating (PP vs. CD)
        if self.alt_training:

            self.cg_mode = False  # Whether we are in PP or in CD mode

            # Set up the loader for alternating evaluation
            self.alt_eval_loader = eval_loader_cg

            # Sanity checks
            if 'alternate_steps' in train_config:
                self.alternate_steps = train_config['alternate_steps']
            else:
                self.alternate_steps = self.args_dict.get("alternate_steps", 8)
            if (self.alternate_steps > (self.args.logging_steps / 2)
                    or self.args.logging_steps % self.alternate_steps != 0 or
                (self.args.logging_steps / self.alternate_steps) % 2 != 0):
                raise ValueError(
                    f"Combination of alternate steps {self.alternate_steps} and logging"
                    f" steps ({self.args.logging_steps}) would break best-model-saving."
                )
            if (self.args.gradient_accumulation_steps > self.alternate_steps
                    or self.args.eval_accumulation_steps > self.alternate_steps
                    or self.alternate_steps %
                    self.args.gradient_accumulation_steps != 0
                    or self.alternate_steps % self.args.eval_accumulation_steps
                    != 0):
                raise ValueError(
                    f"Combination of alternate steps ({self.alternate_steps}) & gradient"
                    f" accumulation steps ({self.args.gradient_accumulation_steps} and "
                    f"{self.args.eval_accumulation_steps}) breaks training logic."
                )

            self.cc_loss_weight = self.args_dict.get("cc_loss_weight", 1)
            self.cc_loss = self.args_dict.get("cc_loss", False)

            # Implement sample-weighted loss function
            self.cg_bce_loss_fn = CrossEntropyLoss(reduction="none")

            if self.cc_loss:
                self.cc_loss_fn = CrossEntropyLoss(reduction="none")
                # This collator is used in the generation task to predict property/ies
                # of the just generated molecule.
                self.cc_collator = TRAIN_COLLATORS["property"](
                    tokenizer=self.tokenizer,
                    property_tokens=self.property_tokens,
                    num_tokens_to_mask=[-1] * len(self.property_tokens),
                    ignore_errors=True,
                )

    def sum_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        return e + num_e

    def overwrite_embed(self, e: torch.Tensor,
                        num_e: torch.Tensor) -> torch.Tensor:
        e[:, :, -self.numerical_encodings_dim:] = num_e
        return e

    def save_attention(self, inputs: torch.Tensor, attention: torch.Tensor):
        """
        Save the attention weights for the current batch.

        Args:
            inputs (torch.Tensor): input_ids
            attention (torch.Tensor): attention tensor

        """

        for idx, a in enumerate(attention):
            for i, aa in enumerate(a):
                np.save(f"batch_{self.counter}_layer_{idx}_tup_{i}",
                        aa.detach().numpy())

        for i, inp in enumerate(inputs):
            tokens = self.tokenizer.convert_ids_to_tokens(inp.tolist())
            with open(f"batch_{self.counter}_sample_{i}.txt", "w") as f:
                f.write(str(tokens))
        self.counter += 1

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

        if self.alt_training and self.cg_mode:
            model_inputs = inputs.copy()
        [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
        outputs = model(**model_inputs)

        # If we enter here, we are in a training step where we are doing cond. gen.
        if self.alt_training and self.cg_mode:

            # Apply conditional generation loss (BCE loss on affected tokens) with
            # custom sample weights to reflect
            #   1) distance of sampled condition to real prop (vanilla CG collator)
            #   2) "negative learning" (Bimodal generator)
            logits = outputs[1].permute(0, 2, 1)
            sample_loss = self.cg_bce_loss_fn(logits,
                                              inputs["labels"]).mean(axis=-1)

            loss = (inputs["sample_weights"] * sample_loss).mean()

            if self.cc_loss:
                # Apply cycle-consistency loss
                cc_loss = self.get_cc_loss(model, inputs, outputs)
                loss += cc_loss * self.cc_loss_weight

            # Overwrite PLM loss with CC loss
            outputs = (loss, *outputs[1:])

        return outputs

    def get_cc_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """Computes self-consistency loss. Receives the model, model inputs for
            conditional generation and the generated molecules.
            Performs a property prediction on the generated molecules and computes the
            loss between the predicted property of the generated molecule and the
            true property.

        Args:
            model (nn.Module): XLNetLMHeadModel
            inputs (Dict[str, torch.Tensor]): Dict of inputs for the model. Should have
                keys 'input_ids' and 'labels' (at least).
            outputs (Tuple[torch.Tensor]): Outputs from the model for the CD generation
                task. Usually a 3-tuple (loss, logits, mems)

        Returns:
            torch.Tensor: Scalar tensor with CC loss.
        """
        # to avoid recursive cycle since `feed_model` is called below
        self.cg_mode = False

        # Extract logits and plain BCE loss
        _loss, logits = outputs[:2]

        # Convert logits to molecules
        predictions = torch.argmax(logits, dim=-1)

        # Combine predictions with labels
        generations = inputs["input_ids"].clone()
        generations[generations == self.tokenizer.mask_token_id] = predictions[
            generations == self.tokenizer.mask_token_id]

        # mask properties (collator normally works on CPU but in this case we pass device)
        cc_input = self.cc_collator.mask_tokens(generations,
                                                device=self.device)
        cc_attention_mask = self.cc_collator.attention_mask(generations)
        cc_input = {
            "input_ids": cc_input[0],
            "perm_mask": cc_input[1],
            "target_mapping": cc_input[2],
            "labels": cc_input[3],
            "attention_mask": cc_attention_mask,
        }

        # Pass through model
        cc_outputs = self.feed_model(model, cc_input)
        cc_loss, cc_logits = cc_outputs[:2]
        cc_logits = cc_logits.permute(0, 2, 1)

        # Compute BCE loss between logits and derived labels
        # Reduction is none so the mean reduces from N x T to a scalar.
        loss = self.cc_loss_fn(cc_logits, cc_input["labels"]).mean()

        assert _loss != loss, f"Losses cant be identical: {loss}"
        return loss

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

    def training_step(
            self, model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            step: int) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        NOTE: Overwritten to
        1) maintain custom embeddings.
        2) maintain alternating optimization modes

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
            step (:obj:`int`):
                The current step to plot some gradients for

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # NOTE: Overwritten to maintain custom embeddings and alternative losses.
        outputs = self.feed_model(model, inputs)
        loss = outputs[0]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.world_size > 1:
            loss = loss.mean(
            )  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss

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
        alt_loader = self.alt_eval_loader if self.alt_training else dataloader

        # NOTE: The below is a mixture of transformers 3.1.0 and 3.4.0. 3.4.0 introduced
        # the CallbackHandler class which effectively requires rewriting large parts
        # of the package.
        disable_tqdm = not self.is_local_process_zero(
        ) or self.args.disable_tqdm
        samples_count = 0

        epoch_pbar = tqdm(dataloader, desc=description, disable=disable_tqdm)
        for step, (inputs, a_inputs) in enumerate(zip(dataloader, alt_loader)):

            epoch_pbar.update(1)
            # To optionally take out keys from the collator dict.
            if pop_and_return_keys:
                for k in pop_and_return_keys:
                    return_collator_data[k].extend(
                        inputs.get(k, torch.Tensor()).tolist())
                    inputs.pop(k, None)

            if self.alt_training:
                self.cg_mode = self.get_cg_mode(step)
                if (self.get_cg_mode(step) > self.get_cg_mode(step - 1)
                        and step % 100 == 0):
                    logger.debug("Switching to CG task")
                if self.cg_mode:
                    inputs = a_inputs

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

    def log(self,
            logs: Dict[str, float],
            iterator: Optional[tqdm] = None) -> None:
        """
        NOTE: Overwritten to save best model alongside some metrics

        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        #super().log(logs, iterator) #FIXME: This was creating bug, need to resolve later

        if "eval_loss" in logs.keys():
            logger.info(f"Evaluation {logs}")
            self.eval_logs.append({"eval_loss": logs["eval_loss"]})
            if "epoch" in logs.keys():
                self.eval_logs[-1].update({
                    "epoch": logs["epoch"],
                    "step": self.global_step
                })

        # Custom logging
        if "loss" in logs.keys():
            # In case of training logging
            if self.epoch is not None:
                logs["epoch"] = self.epoch
                output = {**logs, **{"step": self.global_step}}
                self.logs.append(output)

            # Save new best model
            if logs["loss"] < self.min_loss:
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-best-{self.global_step}"
                output_dir = os.path.join(self.args.output_dir,
                                          checkpoint_folder)
                self.min_loss = logs["loss"]
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_process_zero():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                pd.DataFrame(self.logs).to_csv(
                    os.path.join(output_dir, "training_log.csv"))
                pd.DataFrame(self.eval_logs).to_csv(
                    os.path.join(output_dir, "eval_log.csv"))

                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-best"

                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

    def _rotate_checkpoints(self,
                            use_mtime: bool = False,
                            prefix: str = PREFIX_CHECKPOINT_DIR) -> None:
        """NOTE: Overwritten to enable passing down checkpoint prefix for deletion."""

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        #EA Edit: Don't need to delete older checkpoints? Comment out
        #checkpoints_sorted = self._sorted_checkpoints(
        #    use_mtime=use_mtime, checkpoint_prefix=prefix
        #)

        #if len(checkpoints_sorted) <= self.args.save_total_limit:
        #    return

        #number_of_checkpoints_to_delete = max(
        #    0, len(checkpoints_sorted) - self.args.save_total_limit
        #)
        #checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        #for checkpoint in checkpoints_to_be_deleted:
        #    logger.info(
        #        "Deleting older checkpoint [{}] due to args.save_total_limit".format(
        #            checkpoint
        #        )
        #    )
        #    shutil.rmtree(checkpoint)

    def train(
        self,
        prop_dataloader: DataLoader = None,
        cg_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        model_path: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
    ):
        """
        NOTE: Overwritten to fix a bug in step skipping.

        Main training entry point.

        Args:
            prop_dataloader (:obj:`DataLoader`):
                Instantiated property dataloader.
            cg_dataloader (:obj:`DataLoader`, `optional`):
                Instantiated cg dataloader.
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.

        # Model re-init
        if self.args.model_init is not None:
            print(
                f" ==> self.model_init = {self.model_init}, re-initializing model"
            )
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        if self.tb_writer is not None:
            logs: Dict[str, float] = {}

        # Data loader and number of training steps
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps //
                (len(prop_dataloader) // self.args.gradient_accumulation_steps)
                + 1)
        else:
            t_total = int(
                len(prop_dataloader) // self.args.gradient_accumulation_steps *
                self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(model_path, "optimizer.pt"),
                    map_location=self.device,
                ))
            self.lr_scheduler.load_state_dict(
                torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # Distributed training (should be after apex fp16 initialization)
        #LC Hacks:
        #FIXME: Remove after debugging
        debugging = True
        if debugging:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # END OF CODE TO REMOVE
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            #world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
            #rank=int(os.environ['OMPI_COMM_WORLD_RANK']))
            #world_size=int(os.environ.get('OMPI_COMM_WORLD_SIZE',1)),
            #rank=int(os.environ.get('OMPI_COMM_WORLD_RANK',0)))
            world_size=self.world_size,
            rank=self.rank)
        dist.barrier()

        model = model.to(self.device)
        model = DDP(model,
                    device_ids=[self.device],
                    find_unused_parameters=True)
        #if self.args.local_rank != -1:
        #    model = torch.nn.parallel.DistributedDataParallel(
        #        model,
        #        device_ids=[self.args.local_rank],
        #        output_device=self.args.local_rank,
        #        find_unused_parameters=True,
        #    )
        #END LC Hacks

        #if self.tb_writer is not None:
        #    self.tb_writer.add_text("args", self.args.to_json_string())
        #    self.tb_writer.add_hparams(self.args.to_sanitized_dict(),
        #                               metric_dict={})

        # Train!
        total_train_batch_size = (self.train_batch_size *
                                  self.args.gradient_accumulation_steps *
                                  (torch.distributed.get_world_size()
                                   if self.args.local_rank != -1 else 1))

        logger.info("***** Running training *****")
        logger.info(f"Model device {model.device}")
        logger.info("  Num examples = %d", self.num_examples(prop_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(
                    model_path.split("-")[-1].split(os.path.sep)[0])
                epochs_trained = self.global_step // (
                    len(prop_dataloader) //
                    self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(prop_dataloader) //
                    self.args.gradient_accumulation_steps)

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d",
                            epochs_trained)
                logger.info("  Continuing training from global step %d",
                            self.global_step)
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.device)
        tr_loss_basic = torch.tensor(0.0).to(self.args.device)  # For the non-alt data training loss. Unused if not doing alt training. (For logging)
        tr_loss_alt = torch.tensor(0.0).to(self.args.device)  # For the alt training data/collator
        logging_loss_scalar = 0.0
        logging_loss_scalar_basic = 0.0
        logging_loss_scalar_alt = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero(
        )
        train_pbar = trange(
            epochs_trained,
            int(np.ceil(num_train_epochs)),
            desc="Epoch",
            disable=disable_tqdm,
        )
        # NOTE: Fixing a bug where to few steps are skipped.
        steps_to_skip = (steps_trained_in_current_epoch *
                         self.args.gradient_accumulation_steps)

        # TODO: Consider making separate function for alternate training
        #       Could have `trainer` folder with separate functions
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(prop_dataloader, DataLoader) and isinstance(
                    prop_dataloader.sampler, DistributedSampler):
                prop_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = prop_dataloader

            # Set up alternating iterator (if applicable)
            if cg_dataloader is None:
                # TODO: Remove after debug commit
                #alt_iterator = prop_dataloader
                iterator = epoch_iterator
            else:
                alt_iterator = cg_dataloader
                iterator = zip(epoch_iterator, alt_iterator)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator,
                              desc="Iteration",
                              disable=disable_tqdm)
            t = time()
            steps_to_skip = 0
            #for step, (inputs, a_inputs) in enumerate(
            for step, inputs in enumerate(
                    iterator  #zip(epoch_iterator, alt_iterator)
            ):
                if isinstance(inputs, tuple):
                    inputs, a_inputs = inputs

                # Skip past any already trained steps if resuming training
                if steps_to_skip > 0:
                    steps_to_skip -= 1
                    epoch_pbar.update(1)
                    continue
                logger.debug(f"Step {step}")
                if self.alt_training:
                    self.cg_mode = self.get_cg_mode(step)

                    if self.get_cg_mode(step) > self.get_cg_mode(step - 1):
                        logger.debug(
                            f"Switching to CG task, took {time()-t:.2f}")
                        t = time()
                    elif self.get_cg_mode(step) < self.get_cg_mode(step - 1):
                        logger.debug(
                            f"Switching to PP task, took {time()-t:.2f}")
                        t = time()

                    if self.cg_mode:
                        inputs = a_inputs

                tr_step_loss = self.training_step(model, inputs, step).item()
                tr_loss += tr_step_loss
                #print('debug: self.alt_training', self.alt_training)
                if self.alt_training:
                    if not self.get_cg_mode(step):
                        tr_loss_basic += tr_step_loss
                        print('debug: tr loss basic and tr step loss', tr_loss_alt, tr_step_loss)
                    else:
                        print('debug: tr loss alt and tr step loss', tr_loss_alt, tr_step_loss)
                        tr_loss_alt += tr_step_loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator)
                        <= self.args.gradient_accumulation_steps and
                    (step + 1) == len(epoch_iterator)):

                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.args.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Log gradients to tensorboard
                    if self.args.grad_logging_steps > 0:
                        if self.global_step % self.args.grad_logging_steps == 0:
                            log_gradients_in_model(self.model, self.tb_writer, self.global_step)
                            log_weights_in_model(self.model, self.tb_writer, self.global_step)
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0
                            and self.global_step % self.args.logging_steps
                            == 0) or (self.global_step == 1
                                      and self.args.logging_first_step):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar
                                        ) / self.args.logging_steps
                        if self.alt_training:
                            tr_loss_scalar_basic = tr_loss_basic.item()
                            tr_loss_scalar_alt = tr_loss_alt.item()
                            logs["loss_basic"] = (
                                    tr_loss_scalar_basic - logging_loss_scalar_basic
                            ) / self.args.logging_steps / 2
                            logs["loss_alt"] = (
                                    tr_loss_scalar_alt - logging_loss_scalar_alt
                            ) / self.args.logging_steps / 2
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0])
                        logging_loss_scalar = tr_loss_scalar
                        if self.alt_training:
                            logging_loss_scalar_basic = tr_loss_scalar_basic
                            logging_loss_scalar_alt = tr_loss_scalar_alt

                        self.log(logs)
                        if self.tb_writer is not None:
                            self.tb_writer.add_scalar(
                                f"train/lr", logs["learning_rate"], global_step=self.global_step
                            )
                            self.tb_writer.add_scalar(
                                f"train/loss", logs["loss"], global_step=self.global_step
                            )
                        print(f'==> Epoch {logs["epoch"]}: TRAIN LOSS at step {self.global_step}: {logs["loss"]}')

                    if (self.args.eval_during_training
                            and self.global_step >= self.args.logging_steps
                            and self.global_step % self.args.eval_steps == 0):
                        metrics = self.evaluate(eval_dataloader)
                        ps, rs, ss = self.property_evaluate(eval_dataloader)
                        if self.tb_writer is not None:
                            self.tb_writer.add_scalar(
                                f"eval/loss", metrics["eval_loss"], global_step=self.global_step
                            )
                            self.tb_writer.add_scalar(
                                f"eval/pearson", ps, global_step=self.global_step
                            )
                            self.tb_writer.add_scalar(
                                f"eval/rmse", rs, global_step=self.global_step
                            )
                            self.tb_writer.add_scalar(
                                f"eval/spearman", ss, global_step=self.global_step
                            )
                        # TODO: want to produce output of `eval_language_modeling.py` 
                        # It should also priting the output of eval_lan_model.py

                    if (self.args.save_steps > 0
                            and self.global_step % self.args.save_steps == 0):
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert (
                                model is self.model
                            ), f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = (
                            f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                        output_dir = os.path.join(self.args.output_dir,
                                                  checkpoint_folder)

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints()

                        if self.is_world_process_zero():
                            torch.save(
                                self.optimizer.state_dict(),
                                os.path.join(output_dir, "optimizer.pt"),
                            )
                            torch.save(
                                self.lr_scheduler.state_dict(),
                                os.path.join(output_dir, "scheduler.pt"),
                            )
                gc.collect()

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

            if self.tb_writer is not None:
                if "learning_rate" in logs.keys():
                    self.tb_writer.add_scalar(
                        f"train/lr", logs["learning_rate"], global_step=self.global_step
                    )
                    self.tb_writer.add_scalar(
                        f"train/loss", logs["loss"], global_step=self.global_step
                    )
                    if self.alt_training:
                        self.tb_writer.add_scalar(
                            f"alt_train/basic_loss", logs["loss_basic"], global_step=self.global_step
                        )
                        self.tb_writer.add_scalar(
                            f"alt_train/alt_loss", logs["loss_alt"], global_step=self.global_step
                        )

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        #LC HACK (is this needed?):
        dist.destroy_process_group()
        return TrainOutput(self.global_step,
                           tr_loss.item() / self.global_step, {})

    def get_cg_mode(self, x: int) -> bool:
        """
        For alternating training, we select the mode (PP vs. CG) by alternating every
        `altenate_steps` steps.

        Args:
            x (int): Current step.

        Returns:
            bool: Returns whether or not we are in alternating training mode (i.e.,
                CG optimization) or vanilla training (PP optimization).
        """
        return (x // self.alternate_steps) % 2 == 1

    def property_evaluate(self, eval_dataloader):
        for pidx, prop in enumerate(self.property_tokens):
            property_collator = TRAIN_COLLATORS["property"](
                tokenizer=self.tokenizer,
                property_tokens=[prop],
                num_tokens_to_mask=[-1],
            )
            ps, rs, ss = self.property_prediction(
                eval_dataloader,
                save_path=os.path.join(self.args.output_dir, f'checkpoint-{self.global_step}'))
                #save_path=self.args.output_dir)

            if rs[0] < np.min(self.perfs[pidx, 1, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-rmse-min-{self.global_step}")
                output_dir = os.path.join(self.args.output_dir,
                                          checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_process_zero():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-rmse-min"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(output_dir,
                                         f"best_rmse_{prop[1:-1]}.json")
                with open(save_path, "w") as f:
                    json.dump({"rmse": rs[0], "pearson": ps[0]}, f)
                logger.info(f"New best RMSE: {rs[0]}")

            if ps[0] > np.max(self.perfs[pidx, 0, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-pearson-max-{self.global_step}")
                output_dir = os.path.join(self.args.output_dir,
                                          checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_process_zero():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-pearson-max"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(output_dir,
                                         f"best_pearson_{prop[1:-1]}_.json")
                with open(save_path, "w") as f:
                    json.dump({"rmse": rs[0], "pearson": ps[0]}, f)
                logger.info(f"New best pearson: {ps[0]}")

            if ss[0] > np.max(self.perfs[pidx, 2, :]):
                # Save model
                checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-spearman-max-{self.global_step}")
                output_dir = os.path.join(self.args.output_dir,
                                          checkpoint_folder)
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_process_zero():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-spearman-max"
                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

                save_path = os.path.join(output_dir,
                                         f"best_spearman_{prop[1:-1]}_.json")
                with open(save_path, "w") as f:
                    json.dump(
                        {
                            "rmse": rs[0],
                            "pearson": ps[0],
                            "spearman": ss[0]
                        }, f)
                logger.info(f"New best spearman: {ss[0]}")

            self.perfs[pidx, 0, self.cidx] = ps[0]
            self.perfs[pidx, 1, self.cidx] = rs[0]
            self.perfs[pidx, 2, self.cidx] = ss[0]

        logger.info(
            f"Current prediction performances {self.perfs[:,:,:self.cidx]}")
        self.cidx += 1

        return ps[0], rs[0], ss[0]

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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                               **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        return self.args.local_rank == -1 or dist.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if self.is_world_process_zero():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            logger.info(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

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


    def property_prediction(self,
                            eval_dataloader,
                            eval_params={},
                            save_path=None, 
                            rmse_factor: int = 1):
        """
        Predict the property
        """
    
        self.greedy_search = SEARCH_FACTORY["greedy"]()
        self.sampling_search = SEARCH_FACTORY["sample"](
            temperature=eval_params.get("temperature", 1.0)
        )
        self.beam_search = SEARCH_FACTORY["beam"](
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )

        #eval_dataloader = self.get_custom_dataloader(collator)
    
        # Hack for XLNet tokenizer
        self.tokenizer.real_decoder = self.tokenizer.decode
        #self.tokenizer.decode = self.tokenizer.decode_internal #Original version-EA Edited
        self.tokenizer.decode=self.tokenizer.decode
    
        #for prop in collator.property_tokens:
        for prop in self.property_tokens:
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


    def multi_property_prediction(self,
                                  eval_dataloader,
                                  eval_params={},
                                  save_path=None, 
                                  rmse_factor: int = 1):
        """
        Predict the property
        """
    
        self.greedy_search = SEARCH_FACTORY["greedy"]()
        self.sampling_search = SEARCH_FACTORY["sample"](
            temperature=eval_params.get("temperature", 1.0)
        )
        self.beam_search = SEARCH_FACTORY["beam"](
            temperature=eval_params.get("temperature", 1.0),
            beam_width=eval_params.get("beam_width", 1),
            top_tokens=eval_params.get("beam_top_tokens", 5),
        )

        #eval_dataloader = self.get_custom_dataloader(collator)
        # Forward pass
        logits, label_ids, metrics, input_ids = self.prediction_loop(
            dataloader=eval_dataloader,
            description=f"Predicting {self.property_tokens}",
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

        num_props = len(self.property_tokens)
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

            for i, prop in enumerate(self.property_tokens):
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
            for i, prop in enumerate(self.property_tokens):
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


def setup_collators_for_training(
        train_config: Dict[str, Any], tokenizer: ExpressionBertTokenizer
) -> Tuple[BaseCollator, BaseCollator]:
    alt_training = False
    if 'alternate_tasks' in train_config:
        alt_training = train_config['alternate_tasks']
    # Set up the training strategy (PLM vs. alternating tasks) + loss function
    if alt_training:
        logger.info("Training with alternate tasks")
        # The main collator is the one for property prediction
        data_collator = TRAIN_COLLATORS["property"](
            tokenizer=tokenizer,
            property_tokens=train_config["property_tokens"],
            num_tokens_to_mask=train_config.get("num_tokens_to_mask", None),
            mask_token_order=train_config.get("mask_token_order", None),
        )
        alternating_collator = TRAIN_COLLATORS[train_config["cg_collator"]](
            tokenizer=tokenizer, **train_config["cg_collator_params"])

    else:
        if train_config["task"] == "proponly":
            data_collator = TRAIN_COLLATORS["property"](
                tokenizer=tokenizer,
                property_tokens=train_config["property_tokens"],
                num_tokens_to_mask=train_config.get("num_tokens_to_mask",
                                                    None),
                mask_token_order=train_config.get("mask_token_order", None),
            )
            logger.warning("Training only on property predict")
        elif train_config["task"] == "gen_only":
            data_collator = TRAIN_COLLATORS[train_config["cg_collator"]](
                tokenizer=tokenizer, **train_config["cg_collator_params"])
            logger.warning("Training ONLY on conditional generation")

        else:  #train_config["task"] == "plm":
            logger.info("Training with PLM")
            # Only vanilla PLM training
            data_collator = DataCollatorForPermutationLanguageModeling(
                tokenizer=tokenizer,
                # plm_probability=data_args.plm_probability,
                # max_span_length=data_args.max_span_length,
            )
        alternating_collator = None

    return data_collator, alternating_collator


def read_training_config(args: Arguments):
    with open(args.train_config, 'r') as fp:
        return json.load(fp)


def plot_grad_flow(named_parameters, save_file, ax=None):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    Returns ax
    """

    ave_grads = []
    max_grads= []
    layers = []
    n_good = 0
    n_bad = 0
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            grad = p.grad
            if grad is None:
                #print('No grad for', n)  # seg_embed layers
                continue
            else:
                #print('Got grad for', n)
                pass
            try:
                a = p.grad.abs().mean().item()
                n_good += 1
            except Exception as e:
                #print(f'Could not handle grad for {n}, skipping')
                n_bad += 1
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    print(f'Could handle grads for {n_good}, could not for {n_bad}')
    if n_good > 0:
        pass
    else:
        print(f'Could handle no grads...')
        return
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(max_grads, label='max-gradient')
    ax.hist(ave_grads, label='ave_grads')
    ax.legend()
    fig.savefig(save_file)
    return ax

def log_gradients_in_model(model, tb_logger, step):
    """Log gradients to TB if TB logger is available"""
    if tb_logger is None:
        return
    #logger.info(f'Logging gradients to TB for step {step}')
    log_any = False
    for tag, value in model.named_parameters():
        if value.grad is not None:
            #logger.info(f'logging grad for {tag} for step {step}')
            tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
            log_any = True
    if not log_any:
        print('Did not log any grads')


def log_weights_in_model(model, tb_logger, step):
    """Log weights to TB if TB logger is available"""
    if tb_logger is None:
        return
    for tag, value in model.named_parameters():
        assert value is not None
        tb_logger.add_histogram(tag + "/val", value.cpu(), step)


def setup_trainer(model: Module, args: Arguments, property_tokens: List[str],
                  eval_loader_cg: DataLoader, 
                  tokenizer: ExpressionBertTokenizer, train_config: dict,
                  prediction_loss_only: bool) -> Trainer:
    return Trainer(model, args, property_tokens, eval_loader_cg,
                   tokenizer, train_config, prediction_loss_only)
