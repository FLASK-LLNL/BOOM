################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from boom.data.load_processed_data import load_processed_data
import os


class SMILESDataset:
    def __init__(self, property, split, split_file=None):
        self.property = property.lower()
        if property.lower() not in [
            "density",
            "hof",
            "alpha",
            "cv",
            "g298",
            "gap",
            "h298",
            "homo",
            "lumo",
            "mu",
            "r2",
            "u0",
            "u298",
            "zpve",
        ]:
            raise ValueError(
                "Property must be either 'density' ,'hof','alpha', 'cv','g298', or 'gap'."
            )
        self.split = split.lower()
        if split.lower() not in ["train", "ood", "id"]:
            raise ValueError("Split must be either 'train', 'ood' or 'id'")

        if split_file is None:

            cur_dir = os.getcwd()
            if property.lower() in ["density", "hof"]:
                split_file = os.path.join(cur_dir, "10k_data_with_ood_splits.csv")
            else:
                split_file = os.path.join(
                    cur_dir, "qm9_data_with_ood_splits_with_inchi.csv"
                )
        self.data = load_processed_data(self.property, split_file)

        self.dataset_type = f"{split.lower()}_{property.lower()}"

        key = "smiles" if "smiles" in self.data[self.dataset_type].keys() else "inchi"
        self.smiles = self.data[self.dataset_type][key]
        self.property_values = self.data[self.dataset_type][property.lower()]

        self.mol_props = {}
        for key in self.data[self.dataset_type].keys():
            if key != "smiles":
                self.mol_props[key] = self.data[self.dataset_type][key]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.property_values[idx]


class IDDensityDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("density", "id", split_file)


class IDHoFDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("hof", "id", split_file)


class TrainDensityDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("density", "train", split_file)


class TrainHoFDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("hof", "train", split_file)


class OODDensityDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("density", "ood", split_file)


class OODHoFDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("hof", "ood", split_file)


class OODQM9_alphaDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("alpha", "ood", split_file)


class TrainQM9_alphaDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("alpha", "train", split_file)


class IDQM9_alphaDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("alpha", "id", split_file)


class OODQM9_cvDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("cv", "ood", split_file)


class TrainQM9_cvDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("cv", "train", split_file)


class IDQM9_cvDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("cv", "id", split_file)


class OODQM9_g298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("g298", "ood", split_file)


class TrainQM9_g298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("g298", "train", split_file)


class IDQM9_g298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("g298", "id", split_file)


class OODQM9_gapDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("gap", "ood", split_file)


class TrainQM9_gapDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("gap", "train", split_file)


class IDQM9_gapDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("gap", "id", split_file)


class OODQM9_h298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("h298", "ood", split_file)


class TrainQM9_h298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("h298", "train", split_file)


class IDQM9_h298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("h298", "id", split_file)


class OODQM9_homoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("homo", "ood", split_file)


class TrainQM9_homoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("homo", "train", split_file)


class IDQM9_homoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("homo", "id", split_file)


class OODQM9_lumoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("lumo", "ood", split_file)


class TrainQM9_lumoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("lumo", "train", split_file)


class IDQM9_lumoDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("lumo", "id", split_file)


class OODQM9_muDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("mu", "ood", split_file)


class TrainQM9_muDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("mu", "train", split_file)


class IDQM9_muDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("mu", "id", split_file)


class OODQM9_r2Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("r2", "ood", split_file)


class TrainQM9_r2Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("r2", "train", split_file)


class IDQM9_r2Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("r2", "id", split_file)


class OODQM9_u0Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u0", "ood", split_file)


class TrainQM9_u0Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u0", "train", split_file)


class IDQM9_u0Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u0", "id", split_file)


class TrainQM9_u298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u298", "train", split_file)


class IDQM9_u298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u298", "id", split_file)


class OODQM9_u298Dataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("u298", "ood", split_file)


class TrainQM9_zpveDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("zpve", "train", split_file)


class IDQM9_zpveDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("zpve", "id", split_file)


class OODQM9_zpveDataset(SMILESDataset):
    def __init__(self, split_file=None):
        super().__init__("zpve", "ood", split_file)
