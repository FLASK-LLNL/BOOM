################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import regex as re
from transformers import BertTokenizer

PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>|>|\*|\$|\%[0-9]{2}|[0-9])"


class SMILESTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str = "",
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string


def generate_smiles_vocab(list_of_smiles):
    vocab = set()
    for smiles in list_of_smiles:
        tokens = re.findall(PATTERN, smiles)
        for token in tokens:
            vocab.add(token)
    return vocab


if __name__ == "__main__":
    # Example usage
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    tokenizer = SMILESTokenizer("vocab_smi.txt")
    print(tokenizer.tokenize(smiles))
    print(tokenizer.vocab_size)
