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


def prepare_splits(
    density_file="10k_dft_density_data.csv",
    hof_file="10k_dft_hof_data.csv",
    output_file="10k_dft_data_with_ood_splits.csv",
):
    """
    This function is not for general use, but is placed here for debugging or
    updating generated files. Do not call this function if you would like to use
    BOOM as is presented.
    Args:
        density_file (str): Path to the density data file.
        hof_file (str): Path to the HOF data file.
        output_file (str): Path to the output file.
    """
    random.seed(42)
    # Check if the density file exists
    if not osp.exists(density_file):
        raise FileNotFoundError(f"File {density_file} not found")
    # Check if the hof file exists
    if not osp.exists(hof_file):
        raise FileNotFoundError(f"File {hof_file} not found")
    # Check if the output file exists
    if osp.exists(output_file):
        print(f"File {output_file} already exists. Skipping the preparation.")
        return

    with open(density_file, "r") as f:
        density_data = f.read().splitlines()
    with open(hof_file, "r") as f:
        hof_data = f.read().splitlines()

    dataframe = {}
    num_molecules = 0
    for line in zip(density_data[1:], hof_data[1:]):
        density, hof = line
        smiles_1, density = density.split(",")
        smiles_2, hof = hof.split(",")
        assert smiles_1 == smiles_2
        dataframe[smiles_1] = {
            "density": float(density),
            "hof": float(hof),
        }
        num_molecules += 1
    smiles_strings = []
    density_values = []
    hof_values = []
    for smiles in dataframe:
        smiles_strings.append(smiles)
        density_values.append(dataframe[smiles]["density"])
        hof_values.append(dataframe[smiles]["hof"])

    # Extract the density values
    density_values = np.array(density_values).reshape(-1, 1)
    # Extract the hof values
    hof_values = np.array(hof_values).reshape(-1, 1)

    density_KDE = KernelDensity(kernel="gaussian", bandwidth="scott").fit(
        density_values
    )
    hof_KDE = KernelDensity(kernel="gaussian", bandwidth="scott").fit(hof_values)

    density_kde_scores = density_KDE.score_samples(density_values)
    hof_kde_scores = hof_KDE.score_samples(hof_values)

    kth_density_score_index = np.argpartition(np.exp(density_kde_scores), 1000)[1000]
    kth_hof_score_index = np.argpartition(np.exp(hof_kde_scores), 1000)[1000]
    kth_density_score = np.exp(density_kde_scores[kth_density_score_index])
    kth_hof_score = np.exp(hof_kde_scores[kth_hof_score_index])

    density_kde_scores = np.exp(density_kde_scores)
    hof_kde_scores = np.exp(hof_kde_scores)

    for smiles, hof_score, density_score in zip(
        smiles_strings, hof_kde_scores, density_kde_scores
    ):
        dataframe[smiles]["density_score"] = np.exp(density_score)
        dataframe[smiles]["hof_score"] = np.exp(hof_score)
        if density_score < kth_density_score:
            dataframe[smiles]["density_ood"] = 1
            dataframe[smiles]["density_train"] = 0
            dataframe[smiles]["density_iid"] = 0
        else:
            dataframe[smiles]["density_ood"] = 0
            num = random.random()
            if num < 0.05:
                dataframe[smiles]["density_train"] = 0
                dataframe[smiles]["density_iid"] = 1
            else:
                dataframe[smiles]["density_train"] = 1
                dataframe[smiles]["density_iid"] = 0

        if hof_score < kth_hof_score:
            dataframe[smiles]["hof_ood"] = 1
            dataframe[smiles]["hof_train"] = 0
            dataframe[smiles]["hof_iid"] = 0
        else:
            dataframe[smiles]["hof_ood"] = 0
            num = random.random()
            if num < 0.05:
                dataframe[smiles]["hof_train"] = 0
                dataframe[smiles]["hof_iid"] = 1
            else:
                dataframe[smiles]["hof_train"] = 1
                dataframe[smiles]["hof_iid"] = 0

    # Let's do some sanity checks and make sure no leakage has happened
    ood_density = {"smiles": [], "density": [], "density_score": []}
    ood_hof = {"smiles": [], "hof": [], "hof_score": []}

    iid_density = {"smiles": [], "density": [], "density_score": []}
    iid_hof = {"smiles": [], "hof": [], "hof_score": []}

    train_density = {"smiles": [], "density": [], "density_score": []}
    train_hof = {"smiles": [], "hof": [], "hof_score": []}

    for smiles in dataframe:
        if dataframe[smiles]["density_ood"]:
            ood_density["smiles"].append(smiles)
            ood_density["density"].append(dataframe[smiles]["density"])
            ood_density["density_score"].append(dataframe[smiles]["density_score"])

        if dataframe[smiles]["hof_ood"]:
            ood_hof["smiles"].append(smiles)
            ood_hof["hof"].append(dataframe[smiles]["hof"])
            ood_hof["hof_score"].append(dataframe[smiles]["hof_score"])

        if dataframe[smiles]["density_iid"]:
            iid_density["smiles"].append(smiles)
            iid_density["density"].append(dataframe[smiles]["density"])
            iid_density["density_score"].append(dataframe[smiles]["density_score"])

        if dataframe[smiles]["hof_iid"]:
            iid_hof["smiles"].append(smiles)
            iid_hof["hof"].append(dataframe[smiles]["hof"])
            iid_hof["hof_score"].append(dataframe[smiles]["hof_score"])

        if dataframe[smiles]["density_train"]:
            train_density["smiles"].append(smiles)
            train_density["density"].append(dataframe[smiles]["density"])
            train_density["density_score"].append(dataframe[smiles]["density_score"])

        if dataframe[smiles]["hof_train"]:
            train_hof["smiles"].append(smiles)
            train_hof["hof"].append(dataframe[smiles]["hof"])
            train_hof["hof_score"].append(dataframe[smiles]["hof_score"])

    assert len(ood_density["smiles"]) == 1000, f"{len(ood_density['smiles'])} != 1000"
    assert len(ood_hof["smiles"]) == 1000, f"{len(ood_hof['smiles'])} != 1000"
    assert len(iid_density["smiles"]) > 400
    assert len(iid_hof["smiles"]) > 400

    assert (
        len(train_density["smiles"])
        + len(iid_density["smiles"])
        + len(ood_density["smiles"])
        == num_molecules
    ), f"Total number of molecules is {num_molecules} but the sum of splits isn't equal to it."
    assert (
        len(train_hof["smiles"]) + len(iid_hof["smiles"]) + len(ood_hof["smiles"])
        == num_molecules
    ), f"Total number of molecules is {num_molecules} but the sum of splits isn't equal to it."

    print(
        f"There are {len(ood_density['smiles'])} OOD density samples.",
        f"There are {len(ood_hof['smiles'])} OOD hof samples.",
        f"There are {len(iid_density['smiles'])} IID density samples.",
        f"There are {len(iid_hof['smiles'])} IID hof samples.",
    )

    for smiles in ood_density["smiles"]:
        assert smiles not in iid_density["smiles"]
        assert smiles not in train_density["smiles"]
    for smiles in ood_hof["smiles"]:
        assert smiles not in iid_hof["smiles"]
        assert smiles not in train_hof["smiles"]

    for smiles in iid_density["smiles"]:
        assert smiles not in train_density["smiles"]
    for smiles in iid_hof["smiles"]:
        assert smiles not in train_hof["smiles"]

    # Ok no leakage. Let's write the data to a file

    with open(output_file, "w") as f:
        f.write(
            "smiles,density,hof,density_score,"
            + "hof_score,density_ood,density_train,"
            + "density_iid,hof_ood,hof_train,hof_iid\n"
        )
        for smiles in dataframe:
            f.write(
                f"{smiles},{dataframe[smiles]['density']},"
                + f"{dataframe[smiles]['hof']},"
                + f"{dataframe[smiles]['density_score']},"
                + f"{dataframe[smiles]['hof_score']},"
                + f"{dataframe[smiles]['density_ood']},"
                + f"{dataframe[smiles]['density_train']},"
                + f"{dataframe[smiles]['density_iid']},"
                + f"{dataframe[smiles]['hof_ood']},"
                + f"{dataframe[smiles]['hof_train']},"
                + f"{dataframe[smiles]['hof_iid']}\n"
            )


def download_text_file(url, filename):
    import requests

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        with open(filename, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"File downloaded successfully and saved as {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")


def download_10k_dft_data(data_dir):

    hof_url = "https://raw.githubusercontent.com/FLASK-LLNL/LLNL-10k-Dataset/refs/heads/main/10k_dft_hof_data.csv"
    density_url = "https://raw.githubusercontent.com/FLASK-LLNL/LLNL-10k-Dataset/refs/heads/main/10k_dft_density_data.csv"

    density_fname = osp.join(data_dir, "10k_dft_density_data.csv")
    hof_fname = osp.join(data_dir, "10k_dft_hof_data.csv")
    if not osp.exists(density_fname) and not osp.exists(hof_fname):
        download_text_file(density_url, density_fname)
        download_text_file(hof_url, hof_fname)
    else:
        print(f"File {density_fname} already exists. Skipping the download.")
    return (
        osp.join(data_dir, "10k_dft_density_data.csv"),
        osp.join(data_dir, "10k_dft_hof_data.csv"),
    )


def generate_splits_10k(data_dir, split_file):
    density_file, hof_file = download_10k_dft_data(data_dir)
    prepare_splits(density_file, hof_file, osp.join(data_dir, split_file))
    return None


if __name__ == "__main__":
    print(
        "This script will prepare the splits for the 10k DFT dataset.",
        "The data should already be in the repo so you don't need to call this script.",
        "So it's not recommended to call this script unless you know what you are doing.",
    )

    prepare_splits()
