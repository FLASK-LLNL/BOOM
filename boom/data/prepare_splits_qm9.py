################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import os.path as osp
import numpy as np
from sklearn.neighbors import KernelDensity
import random
from boom.data.qm9_3d_utils import download_dataset_qm9
from tqdm import tqdm
import tarfile
import os
from rdkit.Chem import MolFromSmiles, MolToInchi


def parse_qm9_mol_str(mol_str):
    mol = {}

    xyz_lines = [line.decode("UTF-8") for line in mol_str.readlines()]
    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    _ = xyz_lines[2 : num_atoms + 2]
    mol_freq = xyz_lines[num_atoms + 2]
    mol_smiles_str = xyz_lines[num_atoms + 3]
    mol_inchi = xyz_lines[num_atoms + 4]

    prop_strings = [
        "tag",
        "index",
        "A",
        "B",
        "C",
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
    ]
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props["omega1"] = max(float(omega) for omega in mol_freq.split())

    mol["inchi"] = mol_inchi
    mol["smiles"] = mol_smiles_str.rstrip()
    mol["num_atoms"] = num_atoms
    for key, value in mol_props.items():
        mol[key] = value

    return mol


def download_parse_qm9_dataset(datadir, dataname, save_loc):
    """
    This function is not for general use, but is placed here for debugging or
    updating generated files. Do not call this function if you would like to use
    BOOM as is presented.
    """
    download_dataset_qm9(datadir, dataname)
    tar_file = osp.join(osp.join(datadir, dataname), "dsgdb9nsd.xyz.tar.bz2")

    tar_data = tarfile.open(tar_file, "r")
    files = tar_data.getmembers()
    mol_objs = []
    for f in tqdm(files):
        mol_data = parse_qm9_mol_str(tar_data.extractfile(f))
        mol_objs.append(mol_data)
    tar_data.close()

    props = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "Cv",
    ]
    print("Generating csv files...")

    _fnames = []
    for prop in props:
        file_name = osp.join(save_loc, f"qm9_{prop}.csv")
        _fnames.append(file_name)
        with open(file_name, "w") as f:
            f.write(f"smiles,{prop}\n")
            for mol in tqdm(mol_objs):
                _smiles = mol["smiles"].rstrip().split("\t")[0]
                f.write(f"{_smiles},{mol[prop]}\n")
    return _fnames


def prepare_splits_qm9(
    property_file_array=[
        "qm9_alpha.csv",
        "qm9_cv.csv",
        "qm9_g298.csv",
        "qm9_gap.csv",
        "qm9_h298.csv",
        "qm9_homo.csv",
        "qm9_lumo.csv",
        "qm9_mu.csv",
        "qm9_r2.csv",
        "qm9_u0.csv",
        "qm9_u298.csv",
        "qm9_zpve.csv",
    ],
    output_file="qm9_data_with_ood_splits_with_inchi.csv",
):
    """
    This function is not for general use, but is placed here for debugging or
    updating generated files. Do not call this function if you would like to use
    BOOM as is presented.
    Args:
        property_file_array (list): Array of file names where each file is a
            comma-seperated file with columns SMILES,property. The property name
            is the file name without the csv suffix.
    """
    dataframe = {}
    num_ood_samples = 10000  # number of OOD samples for the dataset
    if osp.exists(output_file):
        print(f"File {output_file} already exists. Skipping the preparation.")
        return

    print("Preparing splits for downloaded data...")
    for property_file in tqdm(property_file_array):
        property_name = osp.basename(property_file).replace(".csv", "")

        random.seed(42)
        # Check if the qm9_alpha file exists
        if not osp.exists(property_file):
            raise FileNotFoundError(f"File {property_file} not found")
        # Check if the output file exists
        with open(property_file, "r") as f:
            property_data = f.read().splitlines()

        num_molecules = 0
        print("Collecting property data for " + property_name)
        for line in property_data[1:]:
            smiles_1, _property = line.split(",")
            if smiles_1 not in dataframe:
                mol = MolFromSmiles(smiles_1)
                inchi = MolToInchi(mol)
                inchi = inchi.replace(",", "$")
                dataframe[smiles_1] = {property_name: float(_property), "inchi": inchi}
            else:
                dataframe[smiles_1][property_name] = float(_property)

            num_molecules += 1

        smiles_strings = []
        property_values = []
        for smiles in dataframe:
            smiles_strings.append(smiles)
            property_values.append(dataframe[smiles][property_name])

        # Extract the alpha values
        property_values = np.array(property_values).reshape(-1, 1).astype(np.float64)

        print("Starting Kernel Density Estimation for " + property_name)
        property_KDE = KernelDensity(kernel="gaussian", bandwidth="scott").fit(
            property_values
        )
        print("Kernel Density Estimation Done!")

        print("Calculating scores for " + property_name)

        property_kde_scores = property_KDE.score_samples(property_values)

        kth_property_score_index = np.argpartition(
            np.exp(property_kde_scores), num_ood_samples
        )[num_ood_samples]
        kth_property_score = np.exp(property_kde_scores[kth_property_score_index])
        ood_indices = np.argpartition(np.exp(property_kde_scores), num_ood_samples)[
            0:num_ood_samples
        ]
        property_kde_scores = np.exp(property_kde_scores)

        counter = 0
        for smiles, property_score in zip(smiles_strings, property_kde_scores):
            dataframe[smiles][property_name + "_score"] = np.exp(property_score)
            if counter in ood_indices:
                dataframe[smiles][property_name + "_ood"] = 1
                dataframe[smiles][property_name + "_train"] = 0
                dataframe[smiles][property_name + "_iid"] = 0
            else:
                dataframe[smiles][property_name + "_ood"] = 0
                num = random.random()
                if num < 0.05:
                    dataframe[smiles][property_name + "_train"] = 0
                    dataframe[smiles][property_name + "_iid"] = 1
                else:
                    dataframe[smiles][property_name + "_train"] = 1
                    dataframe[smiles][property_name + "_iid"] = 0
            counter = counter + 1
        print("Completed " + property_name)
    print("All properties processed!")
    property_names = [osp.basename(filename).replace(".csv", "") for filename in property_file_array]

    with open(output_file, "w") as f:
        # get header for database
        header = "smiles"
        for property_name in property_names:
            header = (
                header
                + ","
                + property_name
                + ","
                + property_name
                + "_score,"
                + property_name
                + "_ood,"
                + property_name
                + "_train,"
                + property_name
                + "_iid"
            )
        header = header + "\n"
        f.write(header)

        for smiles in dataframe:
            line = smiles
            for property_name in property_names:
                line = (
                    line
                    + ","
                    + f"{dataframe[smiles][property_name]},"
                    + f"{dataframe[smiles][property_name+'_score']},"
                    + f"{dataframe[smiles][property_name+'_ood']},"
                    + f"{dataframe[smiles][property_name+'_train']},"
                    + f"{dataframe[smiles][property_name+'_iid']}"
                )
            line += ","
            line += dataframe[smiles]["inchi"]
            line = line + "\n"
            f.write(line)
    print("Done!")


def generate_splits_qm9(data_dir):

    props = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "Cv",
    ]
    # Check if the qm9_alpha file exists

    needs_gen = False
    prop_file_array = []
    for prop in props:
        property_file = osp.join(data_dir, f"qm9_{prop}.csv")
        prop_file_array.append(property_file)
        if not osp.exists(property_file):
            needs_gen = True
            break
    if needs_gen:
        prop_file_array = []
        data_name = "qm9"
        save_loc = osp.join(data_dir, data_name)
        if not osp.exists(save_loc):
            os.makedirs(save_loc)
        prop_file_array = download_parse_qm9_dataset(data_dir, data_name, save_loc)
        print("Splits generated and saved to " + data_dir)

    prepare_splits_qm9(
        property_file_array=prop_file_array,
        output_file=osp.join(data_dir, "qm9_data_with_ood_splits_with_inchi.csv"),
    )


if __name__ == "__main__":
    print(
        "This script will prepare the splits for the QM9  dataset.",
        "The data should already be in the repo so you don't need to call this script.",
        "So it's not recommended to call this script unless you know what you are doing.",
    )

    prepare_splits_qm9()
