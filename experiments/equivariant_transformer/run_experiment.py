################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer import train_model
from dataset import Torch3DDataset, collate_fn
from tqdm import tqdm
import math
import numpy as np
from boom.viz.ParityPlot import OODParityPlot
import argparse


parser = argparse.ArgumentParser(
    description="Run experiments for equivariant transformer"
)

parser.add_argument(
    "--skip-qm9",
    action="store_true",
    help="Skip QM9 dataset",
)

parser.add_argument(
    "--skip-10k",
    action="store_true",
    help="Skip 10k dataset",
)

parser.add_argument(
    "--skip-special",
    action="store_true",
    help="Skip special tasks like R2 and Mu which need different heads",
)


@torch.no_grad()
def test_model(target_name, test_set, model, device, save_output=False, **kwargs):
    model.eval()
    dataset = Torch3DDataset(target_name, test_set)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    criterion = nn.MSELoss(reduction="sum")
    loss = 0
    if save_output:
        preds = []
        gt = []

    for batch in tqdm(dataloader):
        coords, atoms, batch, labels = batch
        coords = coords.to(device)
        batch = batch.to(device)
        atoms = atoms.to(device)
        labels = labels.to(device)
        pred, _ = model(atoms, coords, batch)
        pred = pred

        if save_output:
            preds.append(pred)
            gt.append(labels)
        loss += criterion(pred, labels).item()
    loss = math.sqrt(loss / len(dataset))
    print(loss)

    suffix = kwargs.get("suffix", None)
    target_name = target_name if suffix is None else f"{target_name}_{suffix}"
    if save_output:
        preds = torch.cat(preds, dim=0).squeeze(1).cpu().numpy()
        labels = torch.cat(gt, dim=0).squeeze(1).cpu().numpy()
        np.save(f"results/{target_name}_{test_set}_preds.npy", preds)
        np.save(f"results/{target_name}_{test_set}_labels.npy", labels)
    return preds, labels, dataset.data.smiles


def run_experiment(target, device="cuda", **kwargs):
    property = target.lower()

    is_dipole_moment = property == "mu"
    is_spatial_extent = property == "r2"
    kwargs["is_dipole_moment"] = is_dipole_moment
    kwargs["is_spatial_extent"] = is_spatial_extent
    model = train_model(property, **kwargs)
    model = model.to(device)
    pred_iid_vals, real_iid_vals, iid_smiles = test_model(
        property, "iid", model, device, save_output=True, **kwargs
    )
    pred_ood_vals, real_ood_vals, ood_smiles = test_model(
        property, "ood", model, device, save_output=True, **kwargs
    )
    test_results = {
        "target": target,  # "Density" or "HoF
        "mean_score": -1,  # TODO: match how chemprop calculates it
        "std_score": -1,  # TODO: match how chemprop calculates it
        "iid_smiles": iid_smiles,
        "pred_iid_vals": pred_iid_vals,
        "real_iid_vals": real_iid_vals,
        "ood_smiles": ood_smiles,
        "pred_ood_vals": pred_ood_vals,
        "real_ood_vals": real_ood_vals,
    }
    return test_results


def write_results_csv(results, filename):
    # TODO: Add this to flask_ood as a utility function
    with open(filename, "w") as f:
        f.write("smiles, real_val, pred_val, iid, ood\n")
        for i in range(len(results["iid_smiles"])):
            f.write(
                f"{results['iid_smiles'][i]},{results['real_iid_vals'][i]},"
                + f"{results['pred_iid_vals'][i]},1,0\n"
            )
        for i in range(len(results["ood_smiles"])):
            f.write(
                f"{results['ood_smiles'][i]},{results['real_ood_vals'][i]},"
                + f"{results['pred_ood_vals'][i]},0,1\n"
            )


def process_results(results, plotter, suffix=""):
    target = results["target"]
    pred_iid_vals = results["pred_iid_vals"]
    real_iid_vals = results["real_iid_vals"]
    pred_ood_vals = results["pred_ood_vals"]
    real_ood_vals = results["real_ood_vals"]

    true_labels = {
        "id": real_iid_vals,
        "ood": real_ood_vals,
    }

    fake_labels = {
        "id": pred_iid_vals,
        "ood": pred_ood_vals,
    }

    # Write results to CSV
    write_results_csv(results, f"./results/results_{target}" + suffix + ".csv")
    fig = plotter(true_labels, fake_labels, model_name="ET", title=target)
    fig.savefig(f"./results/{target}_parity_plot" + suffix + ".png")


def main(args):

    properties = []
    reduction = []
    if not args.skip_qm9:
        properties += [
            "HOMO",
            "LUMO",
            "Gap",
            "Alpha",
            "CV",
            "Alpha",
            "Mu",
            "ZPVE",
            "R2",
        ]
        reduction += ["mean", "mean", "mean", "sum", "sum", "sum", "sum", "sum", "sum"]

    if not args.skip_10k:
        properties += ["HoF", "Density"]
        reduction += ["sum", "mean"]

    if not args.skip_special:
        properties = ["Mu"]
        reduction = ["sum"]

    assert len(properties) > 0, "Skipped all data"

    for reduce_op, property in zip(reduction, properties):
        plot_func = partial(OODParityPlot, target_value=property.lower())
        results = run_experiment(
            property,
            lr=1e-4,
            batch_size=64,
            full_model=True,
            add_augmentation=False,
            dropout=0.0,
            reduction_op=reduce_op,
            num_epochs=20,
            random_init=False,
            suffix="_pre_trained",
        )
        process_results(results, plot_func, suffix="_pre_trained")

        results = run_experiment(
            property,
            lr=1e-6,
            batch_size=64,
            full_model=True,
            add_augmentation=False,
            dropout=0.0,
            reduction_op=reduce_op,
            num_epochs=20,
            random_init=True,
            suffix="",
        )
        process_results(results, plot_func, suffix="")


if __name__ == "__main__":
    # TODO: Add argparse to allow for command line arguments
    args = parser.parse_args()
    main(args)
