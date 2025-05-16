################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from flask_ood.datasets.CoordDataset import CoordDataset, QM9CoordDataset
from ase import Atoms
import os

properties = [
    "density",
    "hof",
    "alpha",
    "cv",
    "gap",
    "homo",
    "lumo",
    "mu",
    "r2",
    "zpve",
]

if not os.path.isdir("xyzs"):
    os.mkdir("xyzs")

for property in properties:
    if os.path.exists(f"{property}_ood.xyz"):
        continue

    train: CoordDataset
    iid: CoordDataset
    ood: CoordDataset

    if property in ["hof", "density"]:
        train = CoordDataset(property=property, split="train")
        iid = CoordDataset(property=property, split="id")
        ood = CoordDataset(property=property, split="ood")
    else:
        train = QM9CoordDataset(property=property, split="train")
        iid = QM9CoordDataset(property=property, split="id")
        ood = QM9CoordDataset(property=property, split="ood")

    train_atoms = [
        Atoms(symbols=symbols, positions=positions, info={property: prop_value})
        for (_, positions, symbols, prop_value) in train
    ]
    iid_atoms = [
        Atoms(symbols=symbols, positions=positions, info={property: prop_value})
        for (_, positions, symbols, prop_value) in iid
    ]
    ood_atoms = [
        Atoms(symbols=symbols, positions=positions, info={property: prop_value})
        for (_, positions, symbols, prop_value) in ood
    ]

    for atoms in train_atoms:
        atoms.write(f"xyzs/{property}_train.xyz", append=True)
    for atoms in iid_atoms:
        atoms.write(f"xyzs/{property}_iid.xyz", append=True)
    for atoms in ood_atoms:
        atoms.write(f"xyzs/{property}_ood.xyz", append=True)
