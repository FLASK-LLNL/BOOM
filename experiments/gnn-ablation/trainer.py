################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_model(
    model,
    train_dataset,
    collator,
    target_name,
    model_type,
    num_epochs=100,
    batch_size=64,
    shuffle=True,
    lr=1e-6,
    denoise=False,
    denoise_weight=0.1,
    device="cuda",
    prop_mean=0.0,
    prop_std=1.0,
):
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
    )
    criterion = nn.MSELoss()
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator
    )

    epoch_loss = 0
    iteration = 0
    for _ in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            # To do: fix this so that the models except dictionary of data
            if not isinstance(batch, dict):
                data = [b.to(device) for b in batch]
                pred = model(*data[1:-1])
                loss = criterion(pred, data[-1])

            else:
                # batch is a dict
                h = batch["atom_features"].to(device)
                x = batch["coords"].to(device)
                charges = batch["atom_charges"].to(device)
                edge_index = batch["bond_index"].to(device)
                node_mask = batch["node_mask"].to(device)
                edge_mask = batch["edge_mask"].to(device)
                edge_attr = batch["bond_features"].to(device)
                target = (batch["property"].to(device) - prop_mean) / prop_std
                loss = 0

                pred = model(
                    h=h,
                    x=x,
                    charges=charges,
                    edges=edge_index,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    edge_attr=edge_attr,
                )
                if denoise:
                    denoised_coords = pred[1]
                    pred = pred[0]
                    real_coords = batch["coords"].to(device)
                    # loss = 0.1 * torch.mean(torch.abs(pred - target))
                # breakpoint()
                loss = loss + criterion(pred, target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            iteration += 1
        scheduler.step()
        print(
            f"Iteration [{iteration}]: Train Loss: {epoch_loss / len(dataloader):.4f},"
        )
    torch.save(model.cpu().state_dict(), f"{model_type}_{target_name}_model.pth")

    return model


@torch.no_grad()
def test_model(
    model,
    test_dataset,
    collator,
    target_name,
    model_type,
    device="cuda",
    prop_mean=0.0,
    prop_std=1.0,
):
    model.eval()
    model.to(device)

    dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collator
    )

    criterion = nn.L1Loss(reduction="sum")
    loss = 0
    preds = []
    gt = []

    for batch in tqdm(dataloader):
        if not isinstance(batch, dict):
            batch = [b.to(device) for b in batch]
            pred = model(*batch[1:-1])
            target = batch[-1]
        else:
            h = batch["atom_features"].to(device)
            x = batch["coords"].to(device)
            charges = batch["atom_charges"].to(device)
            edge_index = batch["bond_index"].to(device)
            node_mask = batch["node_mask"].to(device)
            edge_mask = batch["edge_mask"].to(device)
            edge_attr = batch["bond_features"].to(device)
            pred = model(
                h=h,
                x=x,
                charges=charges,
                edges=edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
            )
            pred = pred * prop_std + prop_mean
            if torch.isnan(pred).any():
                print("Nan in prediction")
                raise ValueError("Nan in prediction")
            target = batch["property"].to(device)
        preds.append(pred)
        gt.append(target)
        loss += criterion(pred, target).item()
    l1_loss = loss / len(test_dataset)
    preds = torch.cat(preds, dim=0).squeeze(1).cpu().numpy()
    labels = torch.cat(gt, dim=0).squeeze(1).cpu().numpy()
    np.save(f"results/{model_type}_{target_name}_preds.npy", preds)
    np.save(f"results/{model_type}_{target_name}_labels.npy", labels)
    return preds, labels, test_dataset.smiles
