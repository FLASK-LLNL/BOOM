################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import os.path as osp
import copy
import importlib.resources as pkg_resources
from boom.data.prepare_splits_qm9 import generate_splits_qm9
from boom.data.prepare_splits_10k import generate_splits_10k


def package_data_loader(default_file, split_file=None):
    if split_file is None:
        with pkg_resources.open_text("boom.data", default_file) as f:
            data = f.read().splitlines()
    else:
        if not osp.exists(split_file):
            raise FileNotFoundError(f"File {split_file} not found")
        with open(split_file, "r") as f:
            data = f.read().splitlines()
    return data


def _load_10k_data(split_file):

    # data = package_data_loader("10k_dft_data_with_ood_splits.csv", split_file)
    with open(split_file, "r") as f:
        data = f.read().splitlines()
    subset_dict = {"smiles": [], "density": [], "hof": []}
    data_dict = {
        "train_density": copy.deepcopy(subset_dict),
        "train_hof": copy.deepcopy(subset_dict),
        "ood_density": copy.deepcopy(subset_dict),
        "ood_hof": copy.deepcopy(subset_dict),
        "id_density": copy.deepcopy(subset_dict),
        "id_hof": copy.deepcopy(subset_dict),
    }
    for line in data[1:]:
        values = line.strip().split(",")
        smiles = values[0]
        density = float(values[1])
        hof = float(values[2])
        density_ood = int(values[5])
        density_train = int(values[6])
        hof_ood = int(values[8])
        hof_train = int(values[9])

        density_subset = (
            "train_density"
            if density_train == 1
            else "ood_density" if density_ood == 1 else "id_density"
        )

        hof_subset = (
            "train_hof" if hof_train == 1 else "ood_hof" if hof_ood == 1 else "id_hof"
        )

        data_dict[density_subset]["smiles"].append(smiles)
        data_dict[density_subset]["density"].append(float(density))
        data_dict[density_subset]["hof"].append(float(hof))

        data_dict[hof_subset]["smiles"].append(smiles)
        data_dict[hof_subset]["density"].append(float(density))
        data_dict[hof_subset]["hof"].append(float(hof))
    return data_dict


def _load_qm9_data(target, split_file):
    # data = package_data_loader("qm9_data_with_ood_splits_with_inchi.csv", split_file)
    with open(split_file, "r") as f:
        data = f.read().splitlines()
    subset_dict = {"inchi": [], f"{target}": [], "smiles": []}
    data_dict = {
        f"train_{target}": copy.deepcopy(subset_dict),
        f"ood_{target}": copy.deepcopy(subset_dict),
        f"id_{target}": copy.deepcopy(subset_dict),
    }
    # get column for target values, ood, and train
    value_column = data[0].split(",").index("qm9_" + target)
    ood_column = value_column + 2
    train_column = value_column + 3

    for line in data[1:]:
        values = line.strip().split(",")
        inchi = values[-1].replace("$", ",")
        smiles = values[0]
        target_val = float(values[value_column])
        target_ood = int(values[ood_column])
        target_train = int(values[train_column])

        target_subset = (
            f"train_{target}"
            if target_train == 1
            else f"ood_{target}" if target_ood == 1 else f"id_{target}"
        )
        # inchi = MolToInchi(MolFromSmiles(smiles))

        data_dict[target_subset]["inchi"].append(inchi)
        data_dict[target_subset]["smiles"].append(smiles)
        data_dict[target_subset][target].append(float(target_val))

    return data_dict


def _load_molnet_data(target, split_file=None):
    data = package_data_loader(f"{target}_data_with_ood_splits.csv", split_file)

    subset_dict = {"smiles": [], f"{target}": []}
    data_dict = {
        f"train_{target}": copy.deepcopy(subset_dict),
        f"ood_{target}": copy.deepcopy(subset_dict),
        f"iid_{target}": copy.deepcopy(subset_dict),
    }
    for line in data[1:]:
        values = line.strip().split(",")
        smiles = values[0]
        target_val = float(values[1])
        target_ood = int(values[3])
        target_train = int(values[4])

        target_subset = (
            f"train_{target}"
            if target_train == 1
            else f"ood_{target}" if target_ood == 1 else f"iid_{target}"
        )

        data_dict[target_subset]["smiles"].append(smiles)
        data_dict[target_subset][target].append(float(target_val))
    return data_dict


def load_processed_data(target_name, split_file):

    if target_name in ["density", "hof"]:
        if not osp.exists(split_file):
            data_dir = osp.dirname(split_file)
            generate_splits_10k(data_dir, split_file)
        return _load_10k_data(split_file)
    elif target_name in [
        "alpha",
        "cv",
        "g298",
        "gap",
        "h298",
        "homo",
        "lumo",
        "mu",
        "r2",
        "u298",
        "zpve",
    ]:
        # Check if the split_file exists
        if not osp.exists(split_file):
            data_dir = osp.dirname(split_file)
            generate_splits_qm9(data_dir)
        return _load_qm9_data(target_name, split_file)
    elif target_name in ["freesolv", "esol", "lipo"]:
        return _load_molnet_data(target_name, split_file)
    else:
        raise ValueError(f"Target {target_name} not recognized")


if __name__ == "__main__":
    data_dir = osp.dirname(__file__)
    data = load_processed_data(
        "density", f"{data_dir}/10k_dft_data_with_ood_splits.csv"
    )
    print(data.keys())
