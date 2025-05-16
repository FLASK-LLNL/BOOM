################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from .rt_tokenizers import ExpressionBertTokenizer
from arguments import Arguments


def setup_tokenizer(args: Arguments) -> ExpressionBertTokenizer:
    return ExpressionBertTokenizer.from_pretrained(args.tokenizer)
