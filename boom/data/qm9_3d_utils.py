################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import os
import urllib.request
from os.path import join as join
import logging
from tqdm import tqdm
from rdkit import Chem


def download_dataset_qm9(
    datadir,
    dataname,
    fname="dsgdb9nsd.xyz.tar.bz2",
    splits=None,
    calculate_thermo=True,
    exclude=True,
    cleanup=True,
):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = join(*[datadir, dataname])

    gdb9_url_data = "https://springernature.figshare.com/ndownloader/files/3195389"
    gdb9_tar_data = join(gdb9dir, fname)

    if os.path.exists(gdb9_tar_data):
        logging.info("GDB9 dataset already downloaded!")
        return
    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)
    logging.info(
        "Downloading and processing GDB9 dataset. Output will be in directory: {}.".format(
            gdb9dir
        )
    )

    logging.info("Beginning download of GDB9 dataset!")

    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    logging.info("GDB9 dataset downloaded successfully!")


def xyz_to_mol_data(xyz_data, smiles_list):
    """
    Convert xyz data to mol data.
    """
    xyz_lines = [line.decode("UTF-8") for line in xyz_data.readlines()]
    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2 : num_atoms + 2]
    mol_freq = xyz_lines[num_atoms + 2]
    mol_smiles_str = xyz_lines[num_atoms + 3]
    mol_inchi = xyz_lines[num_atoms + 4]

    atom_positions = []
    atom_elements = []
    for line in mol_xyz:
        atom, x, y, z, _ = line.replace("*^", "e").split()
        atom_positions.append([float(x), float(y), float(z)])
        atom_elements.append(atom)
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

    mol_inchi = mol_inchi.strip().split("\t")[0]
    mol_smiles_list = []

    for smi in set(mol_smiles_str.rstrip().split("\t")):
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol is not None:
            mol_smiles_list.append(Chem.MolToSmiles(rdkit_mol))

    smiles = mol_smiles_list[0]
    found_smiles = False
    for s in mol_smiles_list:
        if s in smiles_list:
            found_smiles = True
            smiles = s
            break
    if not found_smiles:
        logging.warn(
            f"No matching SMILES found in the list for the following smiles {mol_smiles_list} \n {mol_smiles_str}"
        )

    mol_data = {
        "num_atoms": num_atoms,
        "atoms_list": atom_elements,
        "coords_list": atom_positions,
        "smiles": smiles,
        "inchi": mol_inchi,
    }
    return mol_data


def extract_tarfile(fname, outputdir, smiles_list):
    """
    Extract a tarfile to a specified output directory.
    """
    import tarfile

    logging.info(
        "Extracting tarfile: {} to output directory: {}".format(tarfile, outputdir)
    )

    mols = {}
    if tarfile.is_tarfile(fname):
        logging.info("File is a valid tarfile.")
        tardata = tarfile.open(fname, "r")
        file = tardata.getmembers()
        count = 0
        for f in tqdm(file):
            mol_data = xyz_to_mol_data(tardata.extractfile(f), smiles_list)
            mols[mol_data["smiles"]] = mol_data
            mols[mol_data["inchi"]] = mol_data
            count += 1
        tardata.close()
    else:
        logging.error("File is not a valid tarfile. Exiting extraction.")

    logging.info("Extraction complete!")
    return mols


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Define the data directory
    datadir = join(os.getcwd(), "tmp")

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    # Define the dataset name
    dataname = "qm9"

    # Download the dataset
    download_dataset_qm9(datadir, dataname)
    tar_file = join(join(datadir, dataname), "dsgdb9nsd.xyz.tar.bz2")
    with open("qm9_smiles.csv") as f:
        smiles_list = f.readlines()
        smiles_list = [x.strip() for x in smiles_list[1:]]
    smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles_list]
    mols = extract_tarfile(tar_file, join(datadir, dataname), smiles_list)

    breakpoint()
    import pickle

    pickle_file = join(datadir, "qm9_3d.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump(mols, f)
    logging.info(f"Pickle file saved to {pickle_file}!")


if __name__ == "__main__":
    main()
