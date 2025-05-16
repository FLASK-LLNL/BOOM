################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from boom.viz.ParityPlot import OODParityPlot
from ase.io import read
from ase import Atoms
import numpy as np
from mace.calculators import MACECalculator
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

properties = [
    "hof",
    "density",
    "cv",
    "r2",
    "alpha",
    "gap",
    "homo",
    "lumo",
    "mu",
    "zpve",
]
calcs = [
    MACECalculator(
        model_paths=f"{property}.model", default_dtype="float64", device=device
    )
    for property in properties
]

for property, calc in zip(properties, calcs):
    if os.path.exists(f"mace_{property}.png"):
        continue

    iid_atoms = read(f"xyzs/{property}_iid.xyz", index=":")
    ood_atoms = read(f"xyzs/{property}_ood.xyz", index=":")

    true_density_labels = {
        "id": np.array([atoms.info[property] for atoms in iid_atoms]),
        "ood": np.array([atoms.info[property] for atoms in ood_atoms]),
    }

    fake_density_labels = {
        "id": np.array([calc.get_potential_energy(atoms) for atoms in iid_atoms]),
        "ood": np.array([calc.get_potential_energy(atoms) for atoms in ood_atoms]),
    }
    fig = OODParityPlot(
        true_density_labels,
        fake_density_labels,
        model_name="MACE",
        target_value=property,
        title=property,
    )
    fig.savefig(f"mace_{property}.png")
