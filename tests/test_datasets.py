################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import boom.datasets.SMILESDataset


def test_imports():
    _properties = [
        "density",
        "hof",
        "alpha",
        "cv",
        "homo",
        "lumo",
        "mu",
        "r2",
        "zpve",
        "gap",
    ]

    for property in _properties:
        train_dataset = boom.datasets.SMILESDataset.SMILESDataset(property, "train")
        ood_dataset = boom.datasets.SMILESDataset.SMILESDataset(property, "ood")
        id_dataset = boom.datasets.SMILESDataset.SMILESDataset(property, "id")

        assert train_dataset is not None
        assert ood_dataset is not None
        assert id_dataset is not None
    print("All imports are working!")


def test_dataset_size():
    train_density = boom.datasets.SMILESDataset.TrainDensityDataset()
    assert len(train_density) == 8766

    train_hof = boom.datasets.SMILESDataset.TrainHoFDataset()
    assert len(train_hof) == 8783

    iid_density = boom.datasets.SMILESDataset.IDDensityDataset()
    assert len(iid_density) == 440

    iid_hof = boom.datasets.SMILESDataset.IDHoFDataset()
    assert len(iid_hof) == 423

    ood_density = boom.datasets.SMILESDataset.OODDensityDataset()
    assert len(ood_density) == 1000

    ood_hof = boom.datasets.SMILESDataset.OODHoFDataset()
    assert len(ood_hof) == 1000

    print("All dataset sizes are correct!")


if __name__ == "__main__":
    test_imports()
    test_dataset_size()
