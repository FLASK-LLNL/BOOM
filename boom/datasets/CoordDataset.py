################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from boom.data.load_processed_data import load_processed_data
from boom.data.load_processed_3d_data import load_3D_data
import os


class CoordDataset:
    def __init__(
        self,
        property,
        split,
        cached_file="10K_CSD_MOL.pkl",
        split_file=None,
        use_inchi=False,
    ):
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
                "Property must be one of 'density' ,'hof','alpha', 'cv','g298', or 'gap'.'"
            )
        self.split = split.lower()
        if split.lower() not in ["train", "ood", "id"]:
            raise ValueError("Split must be one of 'train', 'ood' or 'id'")

        if split_file is None:

            cur_dir = os.getcwd()
            if property.lower() in ["density", "hof"]:
                split_file = os.path.join(cur_dir, "10k_data_with_ood_splits.csv")
            else:
                split_file = os.path.join(
                    cur_dir, "qm9_data_with_ood_splits_with_inchi.csv"
                )

        self.data = load_3D_data(self.property, cached_file, split_file)

        self.dataset_type = f"{split.lower()}_{property.lower()}"

        data = load_processed_data(property, split_file)

        assert type(data) == dict
        assert type(self.data) == dict

        available_indices = []

        mol_key = "inchi" if ("inchi" in data[self.dataset_type].keys()) else "smiles"
        for i, x in enumerate(data[self.dataset_type][mol_key]):
            if x in self.data.keys():
                available_indices.append(i)

        self.mols = [data[self.dataset_type][mol_key][x] for x in available_indices]
        self.coords = [self.data[x]["coords_list"] for x in self.mols]
        self.atoms = [self.data[x]["atoms_list"] for x in self.mols]

        try:
            self.smiles = [self.data[x]["smiles"] for x in self.mols]

        except:
            self.smiles = [x for x in self.mols]

        self.property_values = [
            data[self.dataset_type][property.lower()][x] for x in available_indices
        ]
        self.mol_props = {}
        for key in data[self.dataset_type].keys():
            if key != "smiles":
                self.mol_props[key] = [
                    data[self.dataset_type][key][x] for x in available_indices
                ]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return (
            self.smiles[idx],
            self.coords[idx],
            self.atoms[idx],
            self.property_values[idx],
        )


class QM9CoordDataset(CoordDataset):
    """ """

    def __init__(
        self,
        property,
        split,
        cached_file="QM9_MOL.pkl",
        split_file=None,
        use_inchi=True,
    ):
        super().__init__(property, split, cached_file, split_file, use_inchi)


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np

    for prop in [
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

        _str = f"Property: {prop} "
        for _split in ["train", "ood", "id"]:
            dataset = CoordDataset(prop, _split, cached_file="QM9_MOL.pkl")
            _str += f"Split: {_split} Number of elements: {len(dataset)} "
            # print(len(dataset))
            for i in tqdm(range(len(dataset))):
                sample = dataset[i]
                coords = np.array(sample[1])

            # print(coords.shape)
        print(_str)
