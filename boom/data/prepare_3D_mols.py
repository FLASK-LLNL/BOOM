################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("rdkit is not installed. Please install it to generate 3D mols.")

import numpy as np
from tqdm import tqdm
from boom.data.load_processed_data import package_data_loader
from boom.data.qm9_3d_utils import download_dataset_qm9, extract_tarfile
import pickle
from os.path import join as join
import os


def print_failure_causes(counts):
    # Not used but can be useful to debug
    for i, k in enumerate(rdDistGeom.EmbedFailureCauses.names):
        print(k, counts[i])
    # in v2022.03.1 two names are missing from `rdDistGeom.EmbedFailureCauses`:
    print("LINEAR_DOUBLE_BOND", counts[i + 1])
    print("BAD_DOUBLE_BOND_STEREO", counts[i + 2])


def conformer(smiles, max_attempts=10000):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        coords = np.array(mol.GetConformer().GetPositions())
        atoms = np.array([x.GetAtomicNum() for x in list(mol.GetAtoms())])
        tensor_dic = {}
        tensor_dic["coords_list"] = coords
        tensor_dic["atoms_list"] = atoms
        return tensor_dic

    except Exception as e:
        print(e)
        print(f"Failed to generate 3D conformer for {smiles}")
        return None


def generate_3D_dataset(cache_file_name, split_file):
    # data = package_data_loader("10k_dft_data_with_ood_splits.csv", split_file)
    with open(split_file, "r") as f:
        data = f.read().splitlines()
    conformer_dic = {}
    for line in tqdm(data[1:]):
        values = line.strip().split(",")
        smiles = values[0]
        conformed_data = conformer(smiles)
        if conformed_data is not None:
            conformer_dic[smiles] = conformed_data

    with open(cache_file_name, "wb") as f:
        pickle.dump(conformer_dic, f)
    return conformer_dic


def retrieve_qm9_dataset(cache_file_name, split_file):
    datadir = join(os.getcwd(), "tmp")
    pickle_file = join(datadir, cache_file_name)

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    # smiles_data = package_data_loader("qm9_data_with_ood_splits_with_inchi.csv")

    with open(split_file, "r") as f:
        data = f.read().splitlines()
    smiles_list = [x.strip().split(",")[0] for x in smiles_data[1:]]
    smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles_list]
    # inchi_list = [Chem.MolToInchi(Chem.MolFromSmiles(x)) for x in smiles_list]

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    dataname = "qm9"
    download_dataset_qm9(datadir, dataname)
    tar_file = join(join(datadir, dataname), "dsgdb9nsd.xyz.tar.bz2")
    mols = extract_tarfile(tar_file, join(datadir, dataname), smiles_list)

    with open(pickle_file, "wb") as f:
        pickle.dump(mols, f)

    with open(pickle_file, "rb") as f:
        return pickle.load(f)
