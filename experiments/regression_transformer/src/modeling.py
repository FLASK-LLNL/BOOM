################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from arguments import Arguments
import json
from pathlib import Path
from safetensors import safe_open
import struct
from tokenization import numerical_encodings as ne

import torch
from torch import nn

from transformers import (AutoConfig, AutoModelForCausalLM, AutoModel,
                          PretrainedConfig, PreTrainedTokenizer)
from transformers import XLNetLMHeadModel
import logging

logger = logging.getLogger(__name__)


def load_or_make_model(args: Arguments, config:PretrainedConfig, ret_pretrained_flag=False) -> nn.Module:
    from_pretrained = False
    try: # Try loading pretrained model
        model = AutoModelForCausalLM.from_pretrained(
            args.model, from_tf=bool(".ckpt" in args.model), 
            config=config, cache_dir=args.cache_dir)
        logger.info("PRETRAINED: Model restored from pretrained weights.")
        from_pretrained = True
    except: # Use random weights when training
        model = AutoModelForCausalLM.from_config(config)
        logger.info("RANDOM WEIGHTS: Training new model from scratch.")
    if not ret_pretrained_flag:
        return model
    else:
        return model, from_pretrained


def get_safetensor_header(file_path) -> dict:
    """Returns metadata from safetensor, where the keys are layer names"""
    with open(file_path, 'rb') as f:
        length_of_header = struct.unpack('<Q', f.read(8))[0]
        header_data = f.read(length_of_header)
        header = json.loads(header_data)
    return header


def check_save_custom_embedding(args: Arguments) -> bool:
    """Returns true if the saved model has CombinedNumericalEmbedding as the
    word_embedding instead of a simple Embedding."""
    model_path = Path(args.model) / 'model.safetensors'
    try:
        headers = get_safetensor_header(model_path)
        is_custom ='transformer.word_embedding.word_embed.weight' in headers
    except: # No pretrained model exists
        is_custom = False
    return is_custom


def setup_model(args: Arguments, tokenizer: PreTrainedTokenizer) -> nn.Module:
    """
    This expects the saved state_dict to be like a "default" XLNet.
    During saving transformer.word_embedding should not be
    CombinedNumericalEmbedding, and the important weight should be copied to
    transformer.word_embedding.weight.
    """
    if check_save_custom_embedding(args):
        return load_from_old(args, tokenizer)
 
    config = AutoConfig.from_pretrained(args.model)
    model = load_or_make_model(args, config)
    model.resize_token_embeddings(len(tokenizer))
    add_numerical_encoding_to_model(config, tokenizer, model)
    return model


def load_from_old(args: Arguments, tokenizer: PreTrainedTokenizer):
    """
    Load with from_pretrained as before, causing the warning msg. Then load
    the model.safetensors file ourself, re-set the necessary variables,
    then call tie weights.
    """
    config = AutoConfig.from_pretrained(args.model)
    model, from_pretrained = load_or_make_model(args, config, ret_pretrained_flag=True)
    model.resize_token_embeddings(len(tokenizer))

    if from_pretrained:
        model_path = Path(args.model) / 'model.safetensors'
        tensors = {}
        with safe_open(model_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        try:
            # On the chance you call this with a not custom embedding model
            weight = tensors['transformer.word_embedding.weight']
        except KeyError:
            weight = tensors['transformer.word_embedding.word_embed.weight']
            model.transformer.word_embedding.weight = nn.Parameter(weight)
            print(f'Setting transformer.word_embedding.weight from transformer.word_embedding.word_embed.weight. You can ignore other warnings about lm_loss, transformer.num_embed.[weight,embedding.weight]. See comments for details')
        # num_embed is always calculated so those can be ignored.
        # Tie weights, which makes the output embedding (lm_loss) weights same as
        # the input embedding (transformer.word_embedding) weights.
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L1717, also see tie_weights.
        model.tie_weights()

    add_numerical_encoding_to_model(config, tokenizer, model)
    # Sanity check
    print(f'These two should have the same weights')
    print('model.transformer.word_embedding.word_embed.weight')
    print(model.transformer.word_embedding.word_embed.weight)
    print('lm_loss')
    print(model.lm_loss.weight)
    return model


def add_numerical_encoding_to_model(config: PretrainedConfig,
                                    tokenizer: PreTrainedTokenizer,
                                    model: torch.nn.Module):
    as_dict = config.to_dict()
    if not (as_dict.get('use_ne', False)
            or as_dict.get('use_numerical_encodings', False)):
        return
    print('Using numerical encoding')
    numerical_encodings_type = as_dict.get("numerical_encodings_type", "float")
    numerical_encodings_format = as_dict.get("numerical_encodings_format",
                                             "sum")
    numerical_encodings_dim = as_dict.get("numerical_encodings_dim", 16)

    if numerical_encodings_format == "concat":
        if numerical_encodings_dim > as_dict["d_model"]:
            raise ValueError(
                "Numerical encoding size cannot exceed embedding size")
    elif numerical_encodings_format == "sum":
        numerical_encodings_dim = as_dict["d_model"]

    else:
        raise ValueError(
            f"Unknown float encoding format {numerical_encodings_format}.")

    NUM_ENCODING_FACTORY = {"float": ne.FloatEncoding, "int": ne.IntEncoding}
    numerical_encoder = NUM_ENCODING_FACTORY[numerical_encodings_type](
        num_embeddings=as_dict["vocab_size"],
        embedding_dim=numerical_encodings_dim,
        vocab=tokenizer.vocab,
        vmax=as_dict.get("vmax", None),
    )

    # Replace embedding component in transformer
    assert isinstance(model, XLNetLMHeadModel)
    new_embed = CombinedNumericalEmbedding(model.transformer.word_embedding,
                                           numerical_encoder,
                                           numerical_encodings_dim,
                                           numerical_encodings_format)
    model.transformer.word_embedding = new_embed


class CombinedNumericalEmbedding(nn.Module):
    """
    A module intended to replace the embeddings in a Regression Transformer
    with combined text/numerical encoding.
    """

    def __init__(self,
                 orig_embed: nn.Embedding,
                 numerical_embed: nn.Embedding,
                 numerical_encodings_dim: int,
                 mode: str = 'concat'):
        super().__init__()
        self.word_embed = orig_embed
        self.num_embed = numerical_embed
        self.mode = mode
        self.numerical_encodings_dim = numerical_encodings_dim

    def forward(self, x):
        emb_word = self.word_embed(x)
        emb_num = self.num_embed(x)
        if self.mode == 'concat':
            # Overwrite embeddings
            emb_word[:, :, -self.numerical_encodings_dim:] = emb_num
            return emb_word
        else:  # 'sum'
            return emb_word + emb_num

