################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from rdkit import Chem
import numpy as np
import os


ChargeDict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

BondDict = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

AtomList = ["H", "C", "N", "O", "F"]
BondList = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def OneHot(atom_vec, max_num=4):
    one_hot = np.zeros((atom_vec.size, max_num))
    one_hot[np.arange(atom_vec.size), atom_vec] = 1
    return one_hot


def Atom_Embedding_Func(mol):
    atom_vec = np.array([AtomList.index(x.GetSymbol()) for x in mol.GetAtoms()])
    return OneHot(atom_vec, len(AtomList))


def Bond_Embedding_Func(bond, max_num=4):
    one_hot = np.zeros((max_num))
    one_hot[BondDict.index(bond.GetBondType())] = 1
    return one_hot


def SmilesToGraph(
    smiles_string: str,
    atom_embedding_func,
    bond_embedding_func,
    removeHs=False,
    dense=False,
):
    _mol = Chem.MolFromSmiles(smiles_string)

    if _mol == None:
        raise ValueError("Inavlid smiles string")

    data = {}
    _mol = _mol if removeHs else Chem.AddHs(_mol)

    num_atoms = _mol.GetNumAtoms()
    data["num_atoms"] = num_atoms

    charges = np.array([ChargeDict[x.GetSymbol()] for x in _mol.GetAtoms()])
    data["charge"] = charges

    atom_mat = atom_embedding_func(_mol)
    data["atoms"] = atom_mat

    if len(_mol.GetBonds()) > 1:
        edge_mat = []
        if dense:
            adj_mat = np.zeros((num_atoms, num_atoms))
            edge_mat = []
        else:
            # Return COO format connectivity
            edge_list = []

        for bond in _mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if dense:
                adj_mat[i, j] = 1
                edge_mat.append(bond_embedding_func(bond))
            else:
                # Double add for undirected graph
                edge_list.append([i, j])
                edge_list.append([j, i])

                edge_mat.append(bond_embedding_func(bond))
                edge_mat.append(bond_embedding_func(bond))

        data["edge_type"] = edge_mat
        if dense:
            data["edge_mat"] = adj_mat
        else:
            data["edge_list"] = edge_list

    return data


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
