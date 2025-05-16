################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from functools import partial
from models import TopologicalGNN, InvariantGNN, EquivariantGNN
from dataset import (
    Torch3DDataset,
    TorchMolDataset,
    batched_3D_collator,
    batched_mol_collator,
)
from boom.viz.ParityPlot import OODParityPlot
from trainer import train_model, test_model
from utils import make_dir
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="topological",
    help="Model to train. Options are 'topological (T)', 'invariant(I)', 'equivariant(E)'",
)
args = parser.parse_args()


def run_experiment(
    target,
    model,
    train_dataset,
    iid_test_dataset,
    ood_test_dataset,
    collator,
    model_type,
    device="cuda",
    lr=1e-6,
    batch_size=64,
    num_epochs=100,
    prop_mean=0.0,
    prop_std=1.0,
    denoise=False,
    denoise_weight=0.1,
):
    property = target.lower()

    model = model.to(device)
    model = train_model(
        model,
        train_dataset,
        collator,
        target_name=property,
        model_type=model_type,
        device=device,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        prop_mean=prop_mean,
        prop_std=prop_std,
        denoise=denoise,
        denoise_weight=denoise_weight,
    )
    model = model.to(device)
    pred_iid_vals, real_iid_vals, iid_smiles = test_model(
        model,
        iid_test_dataset,
        collator=collator,
        target_name=property,
        model_type=model_type,
        device=device,
        prop_mean=prop_mean,
        prop_std=prop_std,
    )
    pred_ood_vals, real_ood_vals, ood_smiles = test_model(
        model,
        ood_test_dataset,
        collator=collator,
        target_name=property,
        model_type=model_type,
        device=device,
        prop_mean=prop_mean,
        prop_std=prop_std,
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


def process_results(results, model_type, plotter):
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
    make_dir("./results")
    write_results_csv(results, f"./results/results_{model_type}_{target}.csv")
    fig = plotter(true_labels, fake_labels, model_type, title=target)
    fig.savefig(f"./results/{model_type}_{target}_parity_plot.png")


def get_norm_constants(dataset) -> tuple[float, float]:
    props = np.array([d[-2] for d in dataset])
    mean = np.mean(props).item()
    std = np.std(props).item()
    return mean, std


def train_topological_gnn():
    for property, reduction, in_node_nf in [
        ("Density", "mean", 5),
        ("HoF", "sum", 5),
        ("Gap", "sum", 5),
        ("HOMO", "mean", 5),
        ("LUMO", "sum", 5),
        ("Alpha", "sum", 5),
        ("CV", "sum", 5),
        ("R2", "sum", 5),
        ("Mu", "sum", 5),
        ("ZPVE", "sum", 5),
    ]:

        train_dataset = TorchMolDataset(
            property, "train", cached_file=f"{property}_train_data.pkl"
        )
        iid_test_dataset = TorchMolDataset(
            property, "id", cached_file=f"{property}_iid_data.pkl"
        )
        ood_test_dataset = TorchMolDataset(
            property, "ood", cached_file=f"{property}_ood_data.pkl"
        )

        prop_mean, prop_std = get_norm_constants(train_dataset)
        model = TopologicalGNN(
            in_edge_nf=4,
            in_node_nf=in_node_nf,
            hidden_nf=64,
            reduce=reduction,
            include_charge=True,
        )
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f"The model has {pytorch_total_params} trainable parameters")

        results = run_experiment(
            target=property,
            model=model,
            train_dataset=train_dataset,
            iid_test_dataset=iid_test_dataset,
            ood_test_dataset=ood_test_dataset,
            collator=batched_mol_collator,
            model_type="TopologicalGNN",
            lr=1e-3,
            batch_size=32,
            num_epochs=20,
            device="cuda",
            prop_mean=prop_mean,
            prop_std=prop_std,
        )
        plot_func = partial(OODParityPlot, target_value=property.lower())
        process_results(results, "TopologicalGNN", plot_func)


def train_invariant_gnn(add_noise=False, noise_amount=0.02):
    in_node_nf = 5
    for property, reduction in [
        ("Density", "mean"),
        ("HoF", "sum"),
        ("HOMO", "mean"),
        ("Gap", "mean"),
        ("LUMO", "mean"),
        ("Alpha", "sum"),
        ("CV", "sum"),
        ("R2", "sum"),
        ("Mu", "sum"),
        ("ZPVE", "sum"),
    ]:
        train_dataset = Torch3DDataset(
            property, "train", add_noise=add_noise, noise_amount=noise_amount
        )
        iid_test_dataset = Torch3DDataset(
            property,
            "iid",
        )
        ood_test_dataset = Torch3DDataset(
            property,
            "ood",
        )
        prop_mean, prop_std = get_norm_constants(train_dataset)
        model = InvariantGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=4,
            hidden_nf=64,
            include_charge=True,
            device="cuda",
            reduce=reduction,
        )
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Property: {property} \t mean: {prop_mean} \t std: {prop_std}")

        print(f"The model has {pytorch_total_params} trainable parameters")

        results = run_experiment(
            target=property,
            model=model,
            train_dataset=train_dataset,
            iid_test_dataset=iid_test_dataset,
            ood_test_dataset=ood_test_dataset,
            collator=batched_3D_collator,
            model_type="InvariantGNN",
            lr=1e-3,
            batch_size=32,
            num_epochs=50,
            device="cuda",
            prop_mean=prop_mean,
            prop_std=prop_std,
        )
        plot_func = partial(OODParityPlot, target_value=property.lower())
        process_results(results, "InvariantGNN", plot_func)


def train_equivariant_gnn(
    add_noise=False, noise_amount=0.05, denoise=True, denoise_weight=0.1
):
    for property, reduction, in_node_nf in [
        ("Density", "mean", 5),
        ("HoF", "sum", 5),
        ("HOMO", "mean", 5),
        ("Gap", "mean", 5),
        ("LUMO", "mean", 5),
        ("Alpha", "sum", 5),
        ("CV", "sum", 5),
        ("R2", "sum", 5),
        ("Mu", "sum", 5),
        ("ZPVE", "sum", 5),
    ]:
        train_dataset = Torch3DDataset(
            property, "train", add_noise=add_noise, noise_amount=noise_amount
        )
        iid_test_dataset = Torch3DDataset(
            property,
            "iid",
        )
        ood_test_dataset = Torch3DDataset(
            property,
            "ood",
        )
        prop_mean, prop_std = get_norm_constants(train_dataset)
        model = EquivariantGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=4,
            hidden_nf=64,
            include_charge=True,
            device="cuda",
            reduce=reduction,
        )
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Property: {property} \t mean: {prop_mean} \t std: {prop_std}")

        print(f"The model has {pytorch_total_params} trainable parameters")

        results = run_experiment(
            target=property,
            model=model,
            train_dataset=train_dataset,
            iid_test_dataset=iid_test_dataset,
            ood_test_dataset=ood_test_dataset,
            collator=batched_3D_collator,
            model_type="EquivariantGNN",
            lr=1e-3,
            batch_size=32,
            num_epochs=100,
            device="cuda",
            prop_mean=prop_mean,
            prop_std=prop_std,
            denoise=False,
            denoise_weight=denoise_weight,
        )
        plot_func = partial(OODParityPlot, target_value=property.lower())
        process_results(results, "EquivariantGNN", plot_func)


def main():
    if args.model.lower() == "topological" or args.model.lower() == "t":
        train_topological_gnn()
    elif args.model.lower() == "invariant" or args.model.lower() == "i":
        train_invariant_gnn()
    elif args.model.lower() == "equivariant" or args.model.lower() == "e":
        train_equivariant_gnn()
    elif args.model.lower() == "all":
        print("Training all models. This may take a while...pp")
        train_topological_gnn()
        train_invariant_gnn()
        train_equivariant_gnn()
    else:
        print(
            "Invalid model type. Please choose from 'topological', 'invariant', or 'equivariant'"
        )


if __name__ == "__main__":
    # TODO: Add argparse to allow for command line arguments
    main()
