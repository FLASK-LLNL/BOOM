################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import argparse
from dataclasses import dataclass


# This class is here to help IDEs figure out the argument types
@dataclass
class Arguments:
    model: str
    tokenizer: str
    train_config: str

    skip_training: bool
    epochs: int
    batch_size: int
    train_data: str

    eval_data: str


def setup_args(parser: argparse.ArgumentParser):
    # Model / tokenizer arguments
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='Path to model file')
    parser.add_argument('--tokenizer',
                        required=True,
                        type=str,
                        help='Path to tokenizer file')
    parser.add_argument('--train-config',
                        required=True,
                        type=str,
                        help='Path to training configuration file')
    parser.add_argument('--model-init',
                        type=str,
                        help='Re-initialize model during `train` function')
    parser.add_argument('--device',
                        default='cuda:0',
                        type=str,
                        help='Default device for model. If gpu unavailable, code updates to cpu')
    parser.add_argument('--cache_dir',
                        default=None,
                        type=str,
                        help='Directory to store pretrained model.')

    # Training arguments
    parser.add_argument('--skip-training',
                        action='store_true',
                        help='If given, skips training',
                        default=False)
    # FIXME: Grab cg collator from train_config
    #parser.add_argument('--cg-collator',
    #                    type=str,
    #                    help='Pass option for alternating collator (`vanilla_cg`, `property`, `multientity_cg`)',
    #                    choices=[None, 'vanilla_cg', 'property', 'multientity_cg'], #TODO: Look into
    #                    default=None)
    parser.add_argument('--train-data',
                        type=str,
                        help='Path to training data file',
                        default=None)
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs to train',
                        default=10)
    parser.add_argument('--per-device-batch-size',
                        type=int,
                        help='Batch size per device',
                        default=8)
    parser.add_argument('--reset-training-loss',
                        action='store_true',
                        help='Reset training loss',
                        default=False)
    parser.add_argument('--alternate-steps',
                        type=int,
                        help='Number of alternate steps in training',
                        default=8)
    parser.add_argument('--cc-loss-weight',
                        type=float,
                        help='Weight (lambda) of cc loss term',
                        default=1)
    parser.add_argument('--cc-loss',
                        action='store_true',
                        help='If given, use cc loss term',
                        default=False)
    parser.add_argument('--max-steps',
                        type=int,
                        help='Maximum number of training steps',
                        default=-1)
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        help='Number of gradient accumulation steps',
                        default=1)
    parser.add_argument('--num-train-epochs',
                        type=int,
                        help='Number of training epochs',
                        default=50000)
    parser.add_argument('--local-rank',
                        type=int,
                        help='Local rank (default -1)',
                        default=-1)
    parser.add_argument('--past-index',
                        type=int,
                        help='Reset past memory state of the beginning of each epoch (when >= 0)',
                        default=-1)
    parser.add_argument('--weight-decay',
                        type=float,
                        help='Weight decay (not applied to `bias` or `LayerNorm.weight` parameters)',
                        default=0.0)
    parser.add_argument('--learning-rate',
                        type=float,
                        help='learning rate for optimizer',
                        default=1e-4)
    parser.add_argument('--lr-scheduler-type',
                        default='linear',
                        type=str,
                        help='Learning rate scheduler type. Options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`')
    parser.add_argument('--warmup-steps',
                        type=int,
                        help='Number of warmup steps',
                        default=0)
    parser.add_argument('--adam-beta1',
                        type=float,
                        help='beta1 parameter for AdamW',
                        default=0.9)
    parser.add_argument('--adam-beta2',
                        type=float,
                        help='beta2 parameter for AdamW',
                        default=0.99)
    parser.add_argument('--adam-epsilon',
                        type=float,
                        help='epsilon parameter for AdamW',
                        default=1e-8)
    parser.add_argument('--max-grad-norm',
                        type=float,
                        help='Maximum gradient norm (for gradient clipping)',
                        default=1.0)
    parser.add_argument('--logging-steps',
                        type=int,
                        help='Number of steps during training run to log',
                        default=200)
    parser.add_argument('--logging-first-step',
                        action='store_true',
                        help='If passed, log first step of training run',
                        default=False)
    parser.add_argument('--no-tboard',
                        action='store_true',
                        help='If passed, log scalar parameters in tensorboard',
                        default=False)
    parser.add_argument('--grad-logging-steps',
                        type=int,
                        help='Number of steps during training run to log gradients to tensorboard',
                        default=-1)
    parser.add_argument('--log-gradients',
                        action='store_true',
                        help='If passed, log gradients in tensorboard',
                        default=False)
    parser.add_argument('--eval-during-training',
                        action='store_true',
                        help='If passed, evaluate model on validation set during training',
                        default=False)
    parser.add_argument('--dataloader-drop-last',
                        action='store_true',
                        help='If passed, drop last incomplete batch if not divisible by the batch size',
                        default=False)
    parser.add_argument('--eval-accumulation-steps',
                        type=int,
                        help='Number of predictions steps to accumulate before moving the tensors to the CPU.',
                        default=-1)
    parser.add_argument('--eval-steps',
                        type=int,
                        help='Interval of training steps between evaluation steps (when --eval-during-training)',
                        default=100)
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='If passed, only evaluate model using eval.py',
                        default=False)
    parser.add_argument('--save-steps',
                        type=int,
                        help='Interval of training steps between model checkpoints',
                        default=1000)
    parser.add_argument('--output-dir',
                        type=str,
                        help='Path to save logs and checkpoints',
                        default='./checkpoints')
    parser.add_argument('--save-total-limit',
                        type=int,
                        help='Limit on number of checkpoints saved',
                        default=20)
    parser.add_argument('--vocab-size',
                        type=int,
                        help='Size of vocabulary',
                        default=-1)
    parser.add_argument('--d-model',
                        type=int,
                        help='Embedding size',
                        default=-1)
    parser.add_argument('--vmax',
                        type=str,
                        help='vmax for numerical encoder')
    parser.add_argument('--model-type',
                        type=str,
                        help='Model type (default: xlnet)',
                        default="xlnet")
    parser.add_argument('--eval-search',
                        type=str,
                        help='Evaluation search type (default: greedy)', #FIXME
                        default="greedy")
    #parser.add_argument('--eval-search-args',
    #                    type=str,
    #                    help='Args for eval search',
    #                    default={}) # Default needs to be empty dict; removing and code will handle it
    parser.add_argument('--save-attention',
                        action='store_true',
                        help='Save output of attention layer',
                        default=False)
    parser.add_argument('--disable-tqdm',
                        action='store_true',
                        help='If passed, disable tqdm',
                        default=False)

    # Evaluation arguments
    parser.add_argument('--eval-data',
                        type=str,
                        help='Path to test data file',
                        default=None)
    parser.add_argument('--verbose-evaluation',
                        action='store_false',
                        help='Pass flag to turn off verbose evaluation of model.',
                        default=True)
    parser.add_argument('--eval-file',
                        type=str,
                        help='Path to file when running eval.py',
                        default=None)
    parser.add_argument('--convert-smiles',
                        action='store_true',
                        help='If given, data is passed in smiles format and needs to be converted to SELFIES',
                        default=False)
    parser.add_argument('--param-path',
                        type=str,
                        help='Path to parameters in .json file when running eval.py',
                        default=None)
    parser.add_argument('--block-size',
                        type=int,
                        help='Size of blocks in tokenizer during language model evaluation',
                        default=-1)


def parse() -> Arguments:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    setup_args(parser)
    return parser.parse_args()
