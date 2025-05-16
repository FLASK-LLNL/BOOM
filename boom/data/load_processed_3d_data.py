################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import os.path as osp
import pickle
from boom.data.prepare_3D_mols import generate_3D_dataset, retrieve_qm9_dataset


def load_3D_data(property, cached_file, split_file=None):
    if osp.exists(cached_file):
        with open(cached_file, "rb") as f:
            return pickle.load(f)
    else:
        if property in ["hof", "density", "freesolv", "esol", "lipo"]:
            return generate_3D_dataset(cached_file, split_file)
        elif property in [
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
            return retrieve_qm9_dataset(cache_file_name=cached_file)
        else:
            ValueError("Other 3D datasets currently not tested")
