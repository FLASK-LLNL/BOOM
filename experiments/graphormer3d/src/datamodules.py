################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch
import numpy as np

####################### Dataset #######################

from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph
from sklearn.preprocessing import OrdinalEncoder
import pickle

class GraphormerDataset_3d(Dataset):
    def __init__(self, smiles, ys, path_3d_file, y_scaler=None):
        super().__init__()

        self.species_encoder = OrdinalEncoder(
            categories = [
                [1, 6, 7, 8],
            ]
        ).fit([[1]])

        with open(path_3d_file, 'rb') as f:
            data_dict = pickle.load(f)

        self.dataset = []
        for smi, y in zip(smiles, ys):
            try:
                pos     = data_dict[smi]['pos']
                species = data_dict[smi]['species']
                species = self.species_encoder.transform(species.reshape(-1, 1)).reshape(-1)
                
                if y_scaler is not None:
                    y = y_scaler.transform(y.reshape(-1, 1))

                data = Data(
                    pos     = torch.tensor(pos,     dtype=torch.float),
                    x       = torch.tensor(species, dtype=torch.long),
                    y       = torch.tensor(y,       dtype=torch.float),
                    smi     = smi,
                )
                self.dataset.append(data)
            except KeyError:
                pass

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx].clone()
        data.edge_index = radius_graph(data.pos, r=1e6)
        i, j  = data.edge_index
        edge_len = torch.linalg.norm(data.pos[j] - data.pos[i], dim=1, keepdim=True)
        data.edge_attr = edge_len
        return data

####################### LightningDataModule #######################

import lightning as L
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

class HighEnergyDataModule_ood_3d(L.LightningDataModule):
    def __init__(self, data_fname, prop, path_3d_file, batch_size=128, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        assert prop in ['den', 'hof']

        self.data_fname   = data_fname
        self.prop         = prop
        self.path_3d_file = path_3d_file
        self.batch_size   = batch_size
        self.num_workers  = num_workers

    def prepare_data(self):
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # Make assignments here (val/train/test split)
        # Called on every process in DDP

        # Read data
        df = pd.read_csv(self.data_fname)
        smi = df.smiles
        den = df.density
        hof = df.hof

        # Split data depending on which y to predict
        if self.prop == 'den':
            train_mask = (df.density_train == 1)
            iid_mask   = (df.density_iid   == 1)
            ood_mask   = (df.density_ood   == 1)
            train_smi, iid_smi, ood_smi = smi[train_mask], smi[iid_mask], smi[ood_mask]
            train_y,   iid_y,   ood_y   = den[train_mask], den[iid_mask], den[ood_mask]
        else:
            train_mask = (df.hof_train == 1)
            iid_mask   = (df.hof_iid   == 1)
            ood_mask   = (df.hof_ood   == 1)
            train_smi, iid_smi, ood_smi = smi[train_mask], smi[iid_mask], smi[ood_mask]
            train_y,   iid_y,   ood_y   = hof[train_mask], hof[iid_mask], hof[ood_mask]
        
        # Standard scaler for y
        train_y = train_y.to_numpy().reshape(-1, 1)
        iid_y   = iid_y.to_numpy().reshape(-1, 1)
        ood_y   = ood_y.to_numpy().reshape(-1, 1)
        self.y_scaler = StandardScaler().fit(train_y)

        # Instantiate data splits
        self.train_set = GraphormerDataset_3d(train_smi, train_y, self.path_3d_file, self.y_scaler)
        self.valid_set = GraphormerDataset_3d(iid_smi,   iid_y,   self.path_3d_file, self.y_scaler)
        self.test_set  = GraphormerDataset_3d(ood_smi,   ood_y,   self.path_3d_file, self.y_scaler)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True,  batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set,  shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def teardown(self):
        # Clean up state after the trainer stops, delete files...
        # Called on every process in DDP
        pass
