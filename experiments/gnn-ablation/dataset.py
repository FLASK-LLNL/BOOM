################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import pickle
import torch
from torch.utils.data import Dataset
from boom.datasets.SMILESDataset import SMILESDataset
from boom.datasets.CoordDataset import CoordDataset
from utils import SmilesToGraph, Atom_Embedding_Func, Bond_Embedding_Func
from typing import Tuple, Union
import os.path as osp
from tqdm import tqdm
import numpy as np


class TorchMolDataset(Dataset):
    """Molecular dataset for topological data.
    This dataset is used for topological GNNs where the only connectivity information
    for molecules is used.
    """

    def __init__(
        self,
        property,
        split,
        remove_Hs=False,
        cached_file=None,
        split_file=None,
    ):
        self.raw_data = SMILESDataset(property, split)
        self.property = property
        self.split = split
        self.data = []
        self.smiles = []
        self.remove_Hs = remove_Hs
        self._process(cached_file)

    def _process(self, cached_file):
        if cached_file is not None and osp.exists(cached_file):
            with open(cached_file, "rb") as f:
                data = pickle.load(f)
                self.data = data[0]
                self.smiles = data[1]
        else:
            print("Processing data...")
            for smiles, prop in tqdm(self.raw_data):
                data = SmilesToGraph(
                    smiles,
                    Atom_Embedding_Func,
                    Bond_Embedding_Func,
                    removeHs=self.remove_Hs,
                )
                self.data.append((data, prop))
                self.smiles.append(smiles)
            if cached_file is not None:
                with open(cached_file, "wb") as f:
                    pickle.dump([self.data, self.smiles], f)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        graph_data, property = self.data[index]
        num_atoms = torch.tensor(graph_data["num_atoms"]).long()

        bond_edge_features = torch.tensor(graph_data["edge_type"]).float()
        bond_edge_index = torch.tensor(graph_data["edge_list"]).long()
        atom_node_features = torch.tensor(graph_data["atoms"]).float()
        atom_charges = torch.tensor(graph_data["charge"]).float()

        property = torch.tensor(property).float()

        return (
            atom_node_features,
            atom_charges,
            bond_edge_features,
            bond_edge_index,
            property,
            num_atoms,
        )


class Torch3DDataset(Dataset):
    def __init__(
        self,
        property,
        split,
        cached_file="10K_CSD_MOL_3D.pkl",
        processed_graph_cache_file="10K_CSD_MOL_GRAPH.pkl",
        split_file=None,
        remove_Hs=False,
        add_noise=False,
        noise_amount=0.1,
    ):

        property = property.lower()
        self.split = split
        self.property = property

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

        processed_graph_cache_file = (
            f"QM9_MOL_GRAPH_{split}_{property}.pkl"
            if property in qm9_properties
            else processed_graph_cache_file
        )

        self.data = CoordDataset(property.lower(), split, cached_file, split_file)
        self.remove_Hs = remove_Hs
        self.smiles = []
        self.processed_data = {}
        self.coords = []
        self.add_noise = add_noise
        self.noise_amount = noise_amount
        self._process(processed_graph_cache_file)

    def __len__(self):
        return len(self.data)

    def _process(self, processed_graph_cache_file):
        if osp.exists(processed_graph_cache_file):
            with open(processed_graph_cache_file, "rb") as f:
                self.processed_data = pickle.load(f)
        else:
            print("Processing data...")

            for smiles in tqdm(self.data.smiles):
                data = SmilesToGraph(
                    smiles,
                    Atom_Embedding_Func,
                    Bond_Embedding_Func,
                    removeHs=self.remove_Hs,
                )
                self.processed_data[smiles] = data
            with open(processed_graph_cache_file, "wb") as f:
                pickle.dump(self.processed_data, f)

    def __getitem__(self, index) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        smiles, coords, atoms, property = self.data[index]
        graph_data = self.processed_data[smiles]

        coords = np.array(coords)
        coords = torch.from_numpy(coords).float()
        property = torch.tensor(property).float()

        if self.add_noise:
            noise = torch.randn_like(coords) * self.noise_amount
            coords = coords + noise
        coords = coords - coords.mean(0)

        num_atoms = torch.tensor(graph_data["num_atoms"]).long()

        bond_edge_features = torch.tensor(np.array(graph_data["edge_type"])).float()
        bond_edge_index = torch.tensor(graph_data["edge_list"]).long()
        atom_node_features = torch.tensor(graph_data["atoms"]).float()
        atom_charges = torch.tensor(graph_data["charge"]).float()

        return (
            coords,
            atom_node_features,
            atom_charges,
            bond_edge_features,
            bond_edge_index,
            property,
            num_atoms,
        )

    def get_smiles(self, index) -> str:
        return self.data[index][0]


def batched_mol_collator(batch):

    num_node_features = batch[0][0].shape[-1]
    num_edge_features = batch[0][2].shape[-1]
    max_nodes = max([x[-1] for x in batch])
    max_edges = max([x[2].shape[0] for x in batch])

    batch_atom_features = torch.zeros((len(batch), max_nodes, num_node_features))
    batch_atom_charges = torch.zeros((len(batch), max_nodes, 1))
    batch_bond_features = torch.zeros((len(batch), max_edges, num_edge_features))
    batch_bond_index = torch.zeros((len(batch), max_edges, 2))
    batch_property = torch.zeros((len(batch), 1))
    batch_num_atoms = torch.zeros((len(batch), 1))
    node_mask = torch.zeros((len(batch), max_nodes, 1))
    edge_mask = torch.zeros((len(batch), max_edges, 1))

    for i, data in enumerate(batch):
        (
            atom_node_features,
            atom_charges,
            bond_edge_features,
            bond_edge_index,
            property,
            num_atoms,
        ) = data

        num_atoms = num_atoms.item()
        num_edges = bond_edge_index.shape[0]
        batch_atom_features[i, :num_atoms] = atom_node_features
        batch_atom_charges[i, :num_atoms, :] = atom_charges.unsqueeze(-1)
        batch_bond_features[i, :num_edges] = bond_edge_features
        batch_bond_index[i, :num_edges] = bond_edge_index
        batch_property[i] = property
        batch_num_atoms[i] = num_atoms
        node_mask[i, :num_atoms, :] = 1
        edge_mask[i, :num_edges, :] = 1

    return (
        batch_num_atoms,
        batch_atom_features,
        batch_bond_index,
        batch_bond_features,
        node_mask,
        edge_mask,
        batch_atom_charges,
        batch_property,
    )


def batched_3D_collator(batch):
    num_node_features = batch[0][1].shape[-1]
    # breakpoint()
    max_nodes = max([x[-1] for x in batch])

    batch_coords = torch.zeros((len(batch), max_nodes, 3))
    graph_batch = [x[1:] for x in batch]

    (
        batch_num_atoms,
        batch_atom_features,
        batch_bond_index,
        batch_bond_features,
        node_mask,
        edge_mask,
        batch_atom_charges,
        batch_property,
    ) = batched_mol_collator(graph_batch)

    for i, data in enumerate(batch):
        coords = data[0]
        num_atoms = int(batch_num_atoms[i].sum().item())
        batch_coords[i, :num_atoms] = coords
    batch_datum = {
        "coords": batch_coords,
        "atom_features": batch_atom_features,
        "atom_charges": batch_atom_charges,
        "bond_features": batch_bond_features,
        "bond_index": batch_bond_index,
        "property": batch_property,
        "num_atoms": batch_num_atoms,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
    }
    return batch_datum


if __name__ == "__main__":
    # Example usage
    dataset = Torch3DDataset("HOMO", "train")

    prop = [x[-2] for x in dataset]
    prop = np.array(prop)

    print(prop.mean(), prop.std())
