################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch
from torch.utils.data import Dataset
from boom.datasets.CoordDataset import CoordDataset
from typing import Tuple, Union
import numpy as np
from statistics import mean, stdev


class Torch3DDataset(Dataset):
    def __init__(
        self,
        property,
        split,
        cached_file="10K_CSD_MOL.pkl",
        processed_graph_cache_file="10K_CSD_MOL_GRAPH.pkl",
        split_file=None,
        add_augmentation=False,
        return_original=False,
        noise_amount=0.1,
    ):
        self.split = split
        self.property = property

        self.add_augmentation = add_augmentation
        self.return_original = return_original
        self.noise_amount = noise_amount

        qm9_properties = [
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
        ]

        cached_file = "QM9_MOL_3D.pkl" if property in qm9_properties else cached_file

        self.process_atoms = property in qm9_properties

        self.data = CoordDataset(property, split, cached_file, split_file)

        processed_graph_cache_file = (
            f"QM9_MOL_GRAPH_{split}_{property}.pkl"
            if property in qm9_properties
            else processed_graph_cache_file
        )
        self.prop_values = [x[-1] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        _, coords, atoms, property = self.data[index]

        if self.process_atoms:
            atom_dict = {"H": 1, "O": 8, "C": 6, "N": 7, "F": 9}
            atoms = np.array([atom_dict[x] for x in atoms])
        coords = np.array(coords)
        coords = torch.from_numpy(coords).float()
        atoms = torch.from_numpy(atoms).long()
        property = torch.tensor(property).float()
        if self.add_augmentation or self.return_original:
            if self.return_original:
                noised_coords = coords + self.noise_amount * torch.randn_like(coords)
                noised_coords = noised_coords - noised_coords.mean(0)
                coords = coords - coords.mean(0)
                return (
                    noised_coords,
                    atoms,
                    property,
                    coords,
                )
            coords += self.noise_amount * torch.randn_like(coords)
        coords = coords - coords.mean(0)

        return coords, atoms, property

    def mean(self):
        return mean(self.prop_values)

    def std(self):
        return stdev(self.prop_values)

    def get_smiles(self, index) -> str:
        return self.data[index][0]


def collate_fn(batch):
    coords = []
    atoms = []
    labels = []
    b = []

    i = 0
    for c, a, l in batch:
        coords.append(c)
        atoms.append(a)
        labels.append(l)
        batch_indices = [i] * c.shape[0]
        b += batch_indices
        i += 1

    return (
        torch.cat(coords, dim=0).float(),
        torch.cat(atoms, dim=0).long(),
        torch.tensor(b, dtype=torch.long),
        torch.as_tensor(labels).unsqueeze(1).float(),
    )


def noised_collate_fn(batch):
    coords = []
    atoms = []
    labels = []
    noised_coords = []
    b = []

    i = 0
    for n, a, l, c in batch:
        noised_coords.append(n)
        coords.append(c)
        atoms.append(a)
        labels.append(l)
        batch_indices = [i] * c.shape[0]
        b += batch_indices
        i += 1

    return (
        torch.cat(noised_coords, dim=0).float(),
        torch.cat(coords, dim=0).float(),
        torch.cat(atoms, dim=0).long(),
        torch.tensor(b, dtype=torch.long),
        torch.as_tensor(labels).unsqueeze(1).float(),
    )


if __name__ == "__main__":
    dataset = Torch3DDataset("homo", "id")
    print(len(dataset))
    print(len(dataset.data.smiles))
    print(dataset.mean())
    print(dataset.std())
