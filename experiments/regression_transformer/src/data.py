################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from arguments import Arguments
from torch.utils.data import Dataset, DataLoader
from transformers import LineByLineTextDataset, PreTrainedTokenizer, TextDataset
from transformers import DataCollatorForPermutationLanguageModeling


# Function copied from Regression Transformer
# NOTE: The [LineByLine]TextDataset datasets are deprecated and will be removed
#       in transformers v5
def get_dataset(
    filepath: str,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    line_by_line: bool = True,
) -> Dataset:
    if line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer,
                                     file_path=filepath,
                                     block_size=block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=filepath,
            block_size=block_size,
        )


def setup_loader(
        args: Arguments,
        tokenizer: PreTrainedTokenizer,
        train_collator, eval_collator) -> tuple[DataLoader, DataLoader]:
    train_loader = eval_loader = None

    if args.train_data:
        print('Training with dataset', args.train_data)
        train_ds = get_dataset(args.train_data, tokenizer, block_size=510)
        train_loader = DataLoader(train_ds,
                                  args.batch_size,
                                  shuffle=True,
                                  collate_fn=train_collator)

    if args.eval_data:
        print('Evaluating on dataset', args.eval_data)
        eval_ds = get_dataset(args.eval_data, tokenizer, block_size=510)
        eval_loader = DataLoader(eval_ds,
                                 args.batch_size,
                                 shuffle=False,
                                 collate_fn=eval_collator)

    return train_loader, eval_loader
