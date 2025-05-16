################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
"""
Language modeling evaluation script
"""
import json
import logging
import torch
import math
import os
import sys
from time import time

import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    set_seed,
)

from tokenization.rt_collators import (
    ConditionalGenerationEvaluationCollator,
    PropertyCollator,
)
import arguments, data, modeling, tokenization, training
from utils.evaluator import Evaluator
#from utils.property_predictors import PREDICT_FACTORY
PREDICT_FACTORY = {}
from tokenization.rt_tokenizers import ExpressionBertTokenizer
from utils.trainer_utils import get_trainer_dict
from utils.utils import (
    #disable_rdkit_logging,
    find_safe_path,
    get_latest_checkpoint,
    get_equispaced_ranges,
)
from selfies import encoder as selfies_encoder

logger = logging.getLogger(__name__)


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def eval_language_model(model, tokenizer, args):
    # Put model on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print(args.param_path)
    with open(args.param_path, "r") as f:
        eval_params = json.load(f)

    param_filename = args.param_path.split("/")[-1].split(".json")[0]

    # Wrap into args to be safe
    args.__dict__.update(eval_params)

    ## NOTE: Results will be stored in model folder
    #model_dir = args.output_dir
    #if "checkpoint" not in model_dir:
    #    model_dir = get_latest_checkpoint(
    #        model_dir, must_contain=eval_params.get("checkpoint-str", "best")
    #    )

    #config_name = os.path.join(model_dir, "config.json")
    #with open(config_name, "r") as f:
    #    model_params = json.load(f)

    #config = AutoConfig.from_pretrained(
    #    config_name, mem_len=model_params.get("mem_len", 1024)
    #)

    #tokenizer = ExpressionBertTokenizer.from_pretrained(model_dir)
    #sep = tokenizer.expression_separator

    #model = AutoModelWithLMHead.from_pretrained(
    #    model_dir, from_tf=bool(".ckpt" in model_dir), config=config
    #)
    #logger.info(f"Model restored from {model_dir}")

    #model.resize_token_embeddings(len(tokenizer))

    sep = tokenizer.expression_separator

    # Load tokenizer config parameters
    tokenizer_config_name = os.path.join(args.model, "tokenizer_config.json")
    with open(tokenizer_config_name, "r") as f:
        tokenizer_params = json.load(f)

    if eval_params.get("block_size", -1) <= 0:
        #eval_params["block_size"] = tokenizer.max_len
        eval_params["block_size"] = tokenizer_params["model_max_length"]
        # Our input block size will be the max possible for the model
    else:
        #eval_params["block_size"] = min(args.block_size, tokenizer.max_len)
        eval_params["block_size"] = min(args.block_size, tokenizer_params["model_max_length"])

    print(f"eval_params: {eval_params}")


    # Check if dataset needs to be converted from smiles to SELFIES
    if args.convert_smiles:
        eval_dirname_list = args.eval_file.split("/")
        eval_fname_list = eval_dirname_list[-1].split(".")
        eval_dirname_list[-1] = f"{eval_fname_list[0]}_TO_SELFIES.{eval_fname_list[-1]}"
        selfies_eval_filename = '/'.join(eval_dirname_list)
        print(f"selfies_eval_filename: {selfies_eval_filename}")
        tmp_eval_filename = selfies_eval_filename.split("/")[-1].split("_")[-1].split(".")[0]
        print(f"tmp_eval_filename: {tmp_eval_filename}")
        if not os.path.isfile(selfies_eval_filename):
            # Modify contents of eval_file but save to different file
            with open(selfies_eval_filename, 'a') as selfies_file: # NEW FILE
                with open(args.eval_file, 'r') as smiles_file: # ORIG FILE
                    while line := smiles_file.readline():
                    #for line in smiles_file:
                        tmp_line = line.rstrip()
                        tmp_list = tmp_line.split("|")
                        assert len(tmp_list)==2
                        tmp_line = f"{tmp_list[0].replace('dens','den')}|{selfies_encoder(tmp_list[1])}"
                        selfies_file.write(f"{tmp_line}\n")
        # Update filename to use for data loading
        args.eval_file = selfies_eval_filename
        
    # Get datasets
    eval_dataset = data.get_dataset(
        args.eval_file,
        block_size=510, #eval_params["block_size"],
        tokenizer=tokenizer,
        line_by_line=eval_params.get("line_by_line", True),
    )
    # FIXME: Remove this (James added it when debugging)
    #eval_loader = DataLoader(eval_dataset,
    #                         args.batch_size,
    #                         shuffle=False,
    #                         collate_fn=eval_collator)
    logger.info(f"Dataset size {len(eval_dataset)}.")
    #


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")
    print(eval_params)
    plm_prob = eval_params["plm_probability"]
    perplexity_plm_prob = eval_params.get("perplexity_plm_prob", 0.2)
    # TODO: THIS COLLATOR DOESN'T APPEAR TO BE USED AT ALL IN Evaluator
    # NOTE: This collator does not provide an attention mask (unlike the refined training
    # collators which prevent attention on padding), however, the model will largely
    # ignore the paddings.
    vanilla_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=perplexity_plm_prob,
        max_span_length=eval_params["max_span_length"],
    )

    # Initialize our Evaluator
    evaluator = Evaluator(
        model=model,
        args=args,
        eval_params=eval_params,
        data_collator=vanilla_collator,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        prediction_loss_only=False,
    )

    result_dir = os.path.join(args.output_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    eval_filename = args.eval_file.split("/")[-1].split("_")[-1].split(".")[0]

    with open(args.eval_file, "r") as f:
        prefix = sep.join(f.readline().split(sep)[:-1]) + sep

    compute_perplexity = False # FIXME: Return perplexity computation after resolving issue with evaluator.evaluate()
    if compute_perplexity:
        # Evaluation
        logger.info("*** Evaluate perplexity ***")

        # Set seed
        if eval_params.get("set_seed", True):
            set_seed(eval_params.get("seed", int(time())))

        eval_output = evaluator.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        results = {"perplexity": perplexity}
        path = os.path.join(
            result_dir, f"{eval_filename}_perplexity_plm_{perplexity_plm_prob}.txt"
        )


        with open(find_safe_path(path), "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    #disable_rdkit_logging()
    property_results = []
    properties = eval_params["property_tokens"]
    orders = eval_params.get("property_token_masking_order", None)
    tokens_to_mask = eval_params.get("property_tokens_to_mask", None)
    conditioning_ranges = eval_params.get(
        "conditioning_range",
        get_equispaced_ranges(
            args.eval_file,
            properties,
            precisions=eval_params.get("property_precisions", [2] * len(properties)),
        ),
    )
    logger.info(f"Conditioning range is {conditioning_ranges}")

    # If the token masking orders is not specified we just evaluate all properties together
    if not orders:
        property_collator = PropertyCollator(
            tokenizer=tokenizer,
            property_tokens=properties,
            num_tokens_to_mask=tokens_to_mask,
            mask_token_order=orders,
        )
        ps, rs = evaluator.multi_property_prediction(
            property_collator,
            save_path=os.path.join(result_dir, eval_filename),
            rmse_factor=eval_params.get("rmse_factor", 1),
        )
    else:

        for prop, order, mask in zip(properties, orders, tokens_to_mask):
            logger.info(f"*** Evaluate property {prop} ***")

            for to_mask in mask:

                # We iteratively make the task harder by masking 1-4 tokens.
                # The order of this is determined by `property_token_masking_order`.
                property_collator = PropertyCollator(
                    tokenizer=tokenizer,
                    property_tokens=[prop],
                    num_tokens_to_mask=[to_mask],
                    mask_token_order=[order],
                )
                print(f"Masking {to_mask} in order {order}")
                ps, rs, ss = evaluator.property_prediction(
                    property_collator,
                    save_path=os.path.join(
                        result_dir, f"{prop[1:-1]}_{eval_filename}_mask_{to_mask}.csv"
                    ),
                    rmse_factor=eval_params.get("rmse_factor", 1),
                )
                for p, r, s, n in zip(ps, rs, ss, ["Greedy", "Sampling", "Beam"]):
                    prop_res_dict = {
                        "prop": prop[1:-1],
                        "pearson": p,
                        "spearman": s,
                        "rmse": r,
                        "search": n,
                        "num_masked": to_mask,
                    }
                    property_results.append(prop_res_dict)

            pd.DataFrame(property_results).to_csv(
                os.path.join(result_dir, f"property_prediction_{eval_filename}.csv")
            )
    for prop, cr in zip(properties, conditioning_ranges):
        logger.info(f"Evaluating conditional generation for {prop} with {cr}")
        conditional_generation_collator = ConditionalGenerationEvaluationCollator(
            tokenizer=tokenizer,
            property_token=prop,
            conditioning_range=cr,
            plm_probability=plm_prob,
            max_span_length=eval_params["max_span_length"],
            entity_to_mask=eval_params.get("entity_to_mask", None),
            entity_separator_token=eval_params.get("entity_separator_token", None),
        )

        # Retrieve the property prediction function from dictionary
        if prop[1:-1] in PREDICT_FACTORY.keys():
            evaluate_fn = PREDICT_FACTORY[prop[1:-1]]
            logger.info(f"Found property predictor for {prop}")
            property_collator = None
        else:
            # If unavailable property is predicted
            evaluate_fn = None

            if orders:
                # In single property prediction mode we just mask the property
                property_collator = PropertyCollator(
                    tokenizer=tokenizer,
                    property_tokens=[prop],
                    num_tokens_to_mask=[-1],
                    mask_token_order=None,
                )
            else:
                # in this case, we use the property predictor from above where all tokens are masked
                pass

            logger.info(
                f"No property predictor for {prop}, using model itself for evaluation"
            )

        evaluator.conditional_generation(
            conditional_generation_collator,
            save_path=os.path.join(
                result_dir,
                f"{prop[1:-1]}_conditional_generation_{param_filename}_{eval_filename}.csv",
            ),
            passed_eval_fn=evaluate_fn,
            property_collator=property_collator,
            denormalize_params=eval_params.get("denormalize", {}).get(prop, None),
            #prefix=prefix, # FIXME: This function does not have `prefix` as an input
        )

    print("  ==> eval_language_model completed, shutting down.")
