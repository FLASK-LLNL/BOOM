################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from functools import partial
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the root mean squared error between y_true and y_pred
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1, "y_true and y_pred must be 1D arrays"
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def ParityPlot(
    y_true: Union[None, np.ndarray] = None,
    y_pred: Union[None, np.ndarray] = None,
    ax: Union[None, Axes] = None,
    label: str = "Predicted",
    dot_size: int = 5,
    color=None,
):
    """
    Plots a parity plot of y_true vs y_pred.
    Use this to standardize the parity plot across the experiments.
    """
    assert (
        y_true is not None and y_pred is not None
    ), "Both y_true and y_pred must be provided"
    assert y_true.ndim == 1 and y_pred.ndim == 1, "y_true and y_pred must be 1D arrays"
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert label in ["ID", "OOD"], "label must be either 'ID' or 'OOD'"

    _rmse = rmse(y_true, y_pred)

    label = f"{label} ({_rmse:.4f})"

    if color is not None:
        sns.scatterplot(
            x=y_true,
            y=y_pred,
            ax=ax,
            alpha=1,
            size=5,
            legend=False,
            label=label,
            color=color,
        )
    else:
        sns.scatterplot(
            x=y_true, y=y_pred, ax=ax, alpha=1, size=5, legend=False, label=label
        )


def OODParityPlot(
    true_labels: dict,
    pred_labels: dict,
    model_name,
    target_value: str = "target",
    title: str = "Experiment Name Placeholder",
    dot_size: int = 5,
    line_width: float = 3.5,
    line_alpha: float = 0.5,
) -> Figure:
    """
    Plots party plots for IID and OOD test samples
    """
    assert (
        "id" in true_labels and "ood" in true_labels
    ), "true_labels must contain 'id' and 'ood' keys"
    assert (
        "id" in pred_labels and "ood" in pred_labels
    ), "pred_labels must contain 'id' and 'ood' keys"
    assert len(true_labels["id"]) == len(
        pred_labels["id"]
    ), "Number of ID samples in true_labels and pred_labels must be the same"
    assert len(true_labels["ood"]) == len(
        pred_labels["ood"]
    ), "Number of OOD samples in true_labels and pred_labels must be the same"

    assert target_value in [
        "density",
        "hof",
        "gap",
        "alpha",
        "cv",
        "g298",
        "h298",
        "lumo",
        "homo",
        "mu",
        "r2",
        "u0",
        "u298",
        "zpve",
    ], "target_value must be either 'density', 'hof', 'alpha','cv', or  'gap'."

    # Create a figure and axis
    plt.rc("font", weight="bold")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    extent_dict = {
        "density": [1, 2],
        "hof": [-1000, 400],
        "alpha": [0, 200],
        "cv": [0, 50],
        "gap": [0, 0.7],
        "g298": [-700, 0],
        "homo": [-0.45, 0],
        "lumo": [-0.2, 0.25],
        "mu": [0, 30],
        "r2": [0, 3500],
        "zpve": [0, 0.35],
    }

    extent = extent_dict[target_value] if target_value in extent_dict else [0, 1]
    ax.plot(
        extent,
        extent,
        "r--",
        linewidth=line_width,
        alpha=1,
        label="Perfect Prediction",
        zorder=0,
    )

    # Plot the parity plot for OOD samples
    ParityPlot(
        y_true=true_labels["ood"],
        y_pred=pred_labels["ood"],
        ax=ax,
        label="OOD",
        dot_size=dot_size,
        color="darkblue",
    )

    # Plot the parity plot for IID samples
    ParityPlot(
        y_true=true_labels["id"],
        y_pred=pred_labels["id"],
        ax=ax,
        label="ID",
        dot_size=dot_size,
        color="darkorange",
    )

    ax.set_title(model_name + " - " + title, weight="bold", fontsize="x-large")

    target_value_dict = {
        "density": "Density",
        "hof": "HoF",
        "alpha": "Alpha",
        "cv": "Cv",
        "g298": "G298",
        "gap": "Band Gap",
        "h298": "H298",
        "homo": "HOMO",
        "lumo": "LUMO",
        "mu": "Mu",
        "r2": "R2",
        "u0": "U0",
        "u298": "U298",
        "zpve": "ZPVE",
    }
    target_value = target_value_dict[target_value]
    ax.set_xlabel(f"True {target_value}", weight="bold", fontsize="x-large")
    ax.xaxis.set_tick_params(labelsize="large")
    ax.yaxis.set_tick_params(labelsize="large")
    ax.set_ylabel(f"Predicted {target_value}", weight="bold", fontsize="x-large")
    ax.legend(
        prop={"weight": "bold"},
    )

    plt.tight_layout()
    # plt.rc(
    #     "axes",
    #     labelsize="x-large",
    #     labelweight="bold",
    #     titlesize="x-large",
    #     titleweight="bold",
    # )
    return fig


# 10k Datasets
DensityOODParityPlot = partial(OODParityPlot, target_value="density")
HoFOODParityPlot = partial(OODParityPlot, target_value="hof")

# QM9 Datasets
AlphaOODParityPlot = partial(OODParityPlot, target_value="alpha")
CvOODParityPlot = partial(OODParityPlot, target_value="cv")
G298OODParityPlot = partial(OODParityPlot, target_value="g298")
GapOODParityPlot = partial(OODParityPlot, target_value="gap")
H298OODParityPlot = partial(OODParityPlot, target_value="h298")
HomoOODParityPlot = partial(OODParityPlot, target_value="homo")
LumoOODParityPlot = partial(OODParityPlot, target_value="lumo")
MuOODParityPlot = partial(OODParityPlot, target_value="mu")
R2OODParityPlot = partial(OODParityPlot, target_value="r2")
U0OODParityPlot = partial(OODParityPlot, target_value="u0")
U298OODParityPlot = partial(OODParityPlot, target_value="u298")
ZpveOODParityPlot = partial(OODParityPlot, target_value="zpve")

if __name__ == "__main__":
    # Test the parity plot
    true_iid = 1.8 + 0.05 * np.random.randn(10)
    pred_iid = true_iid + 0.1 * np.random.randn(10)

    true_ood = 1.5 + 0.1 * np.random.randn(100)
    pred_ood = true_ood + 0.1 * np.random.randn(100)

    true_labels = {
        "id": true_iid,
        "ood": true_ood,
    }
    pred_labels = {
        "id": pred_iid,
        "ood": pred_ood,
    }
    fig = OODParityPlot(
        true_labels=true_labels,
        pred_labels=pred_labels,
        model_name="Test Model",
        target_value="density",
        title="Test Parity Plot",
    )

    save_path = "test_parity_plot.png"
    fig.savefig(save_path)
