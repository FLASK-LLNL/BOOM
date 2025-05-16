################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import numpy as np
import matplotlib.pyplot as plt


def plot_dataset(dataset_name="qm9"):
    available_datasets = ["qm9", "10k_dft", "all"]
    assert (
        dataset_name in available_datasets
    ), 'Dataset name must be "qm9","10k_dft" or "all".'
    num_bins = 30
    if dataset_name != "all":
        dataset_name_array = [dataset_name]
    else:
        dataset_name_array = available_datasets
        dataset_name_array.remove("all")

    for dataset_name in dataset_name_array:
        if dataset_name != "10k_dft":
            data = np.loadtxt(
                dataset_name + "_data_with_ood_splits_with_inchi.csv",
                dtype=str,
                delimiter=",",
                comments=None,
            )
            num_properties = int((np.shape(data[0])[0] - 1) / 5)

            for prop in range(num_properties):
                prop_name = data[0][prop * 5 + 1]
                ood_column = (prop * 5) + 3
                train_column = (prop * 5) + 4
                iid_column = (prop * 5) + 5
                ood_indices = np.where(data[:, ood_column] == "1")[0]
                iid_indices = np.where(data[:, iid_column] == "1")[0]
                train_indices = np.where(data[:, train_column] == "1")[0]
                prop_index = (prop * 5) + 1
                data_min, data_max = min(
                    np.array(data[:, prop_index][1:], dtype=float)
                ), max(np.array(data[:, prop_index][1:], dtype=float))
                bins = np.linspace(data_min, data_max, num_bins)

                plt.hist(
                    np.array(data[:, prop_index][train_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="Train (N=" + str(len(train_indices)) + ")",
                )
                plt.hist(
                    np.array(data[:, prop_index][iid_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="IID Test (N=" + str(len(iid_indices)) + ")",
                )
                plt.hist(
                    np.array(data[:, prop_index][ood_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="OOD Test (N=" + str(len(ood_indices)) + ")",
                )
                # plt.hist(np.array(data[:,prop_index][1:],dtype=float),bins=bins,alpha=0.3,label='All',fill=None)

                plt.title("Histogram of Property: " + str(prop_name), fontsize=16)
                plt.ylabel("# of Molecules", fontsize=14)
                plt.xlabel(prop_name + " Value", fontsize=14)
                plt.legend()
                plt.savefig(
                    "viz/" + dataset_name + "_" + prop_name + "_histogram.png",
                )
                plt.savefig(
                    "viz/" + dataset_name + "_" + prop_name + "_histogram.pdf",
                )
                plt.close()

        elif dataset_name == "10k_dft":
            data = np.loadtxt(
                dataset_name + "_data_with_ood_splits.csv",
                dtype=str,
                delimiter=",",
                comments=None,
            )
            num_properties = 2
            for prop in range(num_properties):
                prop_name = data[0][prop + 1]
                ood_column = np.where(data[0] == prop_name + "_ood")[0][0]
                iid_column = np.where(data[0] == prop_name + "_iid")[0][0]
                train_column = np.where(data[0] == prop_name + "_train")[0][0]
                ood_indices = np.where(data[:, ood_column] == "1")[0]
                iid_indices = np.where(data[:, iid_column] == "1")[0]
                train_indices = np.where(data[:, train_column] == "1")[0]
                prop_index = prop + 1
                data_min, data_max = min(
                    np.array(data[:, prop_index][1:], dtype=float)
                ), max(np.array(data[:, prop_index][1:], dtype=float))
                bins = np.linspace(data_min, data_max, num_bins)

                plt.hist(
                    np.array(data[:, prop_index][train_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="Train (N=" + str(len(train_indices)) + ")",
                )
                plt.hist(
                    np.array(data[:, prop_index][iid_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="IID Test (N=" + str(len(iid_indices)) + ")",
                )
                plt.hist(
                    np.array(data[:, prop_index][ood_indices], dtype=float),
                    bins=bins,
                    alpha=0.3,
                    label="OOD Test (N=" + str(len(ood_indices)) + ")",
                )
                # plt.hist(np.array(data[:,prop_index][1:],dtype=float),bins=bins,alpha=0.3,label='All',fill=None)

                plt.title("Histogram of Property: " + str(prop_name), fontsize=16)
                plt.ylabel("# of Molecules", fontsize=14)
                plt.xlabel(prop_name + " Value", fontsize=14)
                plt.legend()
                plt.savefig(
                    "viz/" + dataset_name + "_" + prop_name + "_histogram.png",
                )
                plt.savefig(
                    "viz/" + dataset_name + "_" + prop_name + "_histogram.pdf",
                )
                plt.close()


if __name__ == "__main__":
    plot_dataset("all")
