################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from datasets import load_dataset, Dataset
from boom.datasets.SMILESDataset import SMILESDataset
import numpy as np


def load_data(
    dataset_name: str,
    split: str = "train",
):
    return 0


class BOOMHFDatasetWrapper:
    def __init__(self, property: str, tokenizer, normalize: bool = True):
        self.dataset_name = property
        self.raw_train_dataset = SMILESDataset(
            property=property,
            split="train",
        )
        self.raw_id_test_dataset = SMILESDataset(
            property=property,
            split="id",
        )
        self.raw_ood_test_dataset = SMILESDataset(
            property=property,
            split="ood",
        )
        self.normalize = normalize
        self.tokenizer = tokenizer
        self._set_statistics()
        self._process_dataset()

    def _set_statistics(self):
        if self.normalize:
            prop_vals = np.array(self.raw_train_dataset.property_values)
            self.mean = prop_vals.mean()
            self.std = prop_vals.std()

    def _to_dict(self, dataset):
        _return_dict = {}
        _return_dict["smiles"] = dataset.smiles
        _return_dict["labels"] = np.array(dataset.property_values)
        if self.normalize:
            _return_dict["labels"] = (_return_dict["labels"] - self.mean) / self.std
        return _return_dict

    def _process_dataset(self):

        def tokenize_function(examples):
            return self.tokenizer(
                examples["smiles"],
                padding="max_length",
                truncation=True,
                max_length=256,
            )

        pre_tokenized_train_dataset = Dataset.from_dict(
            self._to_dict(self.raw_train_dataset)
        )
        tokenized_train_dataset = pre_tokenized_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["smiles"],
        )
        self._train_dataset = tokenized_train_dataset

        pre_tokenized_id_test_dataset = Dataset.from_dict(
            self._to_dict(self.raw_id_test_dataset)
        )
        tokenized_id_test_dataset = pre_tokenized_id_test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["smiles"],
        )
        self._id_test_dataset = tokenized_id_test_dataset
        pre_tokenized_ood_test_dataset = Dataset.from_dict(
            self._to_dict(self.raw_ood_test_dataset)
        )
        tokenized_ood_test_dataset = pre_tokenized_ood_test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["smiles"],
        )
        self._ood_test_dataset = tokenized_ood_test_dataset

    def train_dataset(self):
        return self._train_dataset

    def id_test_dataset(self):
        return self._id_test_dataset

    def ood_test_dataset(self):
        return self._ood_test_dataset

    def denormalize(self, _array):
        if self.normalize:
            return _array * self.std + self.mean
        else:
            return _array


if __name__ == "__main__":
    # Example usage
    property = "density"
    from tokenizer import SMILESTokenizer

    tokenizer = SMILESTokenizer(
        vocab_file="vocab_smi.txt",
    )
    dataset = BOOMHFDatasetWrapper(property, tokenizer=tokenizer)
    print(dataset.train_dataset())
    print(dataset.id_test_dataset())
    print(dataset.ood_test_dataset())

    print(dataset.train_dataset()[0])
