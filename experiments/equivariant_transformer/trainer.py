################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataset import Torch3DDataset, collate_fn, noised_collate_fn
from torch.utils.data import DataLoader
from model import ETModel, Loss_EMA
import torch.nn as nn
from torch.utils.data import random_split


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    target_name,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    lr=1e-6,
    device="cuda",
    num_epochs=500,
    add_augmentation=False,
    self_supervised=False,
    full_model=False,
    reduction_op="mean",
    dropout: float = 0,
    random_init=False,
    suffix=None,
    is_dipole_moment=False,
    is_spatial_extent=False,
):
    c_func = noised_collate_fn if self_supervised else collate_fn
    dataset = Torch3DDataset(
        target_name,
        "train",
        add_augmentation=add_augmentation,
        return_original=self_supervised,
        noise_amount=0.5,
    )

    training_samples = int(0.9 * len(dataset))
    validation_samples = len(dataset) - training_samples
    training_dataset, validation_dataset = random_split(
        dataset, [training_samples, validation_samples]
    )
    training_mean = dataset.mean()
    training_std = dataset.std()

    dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=c_func
    )
    # To do: Add validation split to verify model performance
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=c_func
    )
    model = ETModel(
        "pre-trained.ckpt",
        device,
        dropout=dropout,
        self_supervised=self_supervised,
        full_model=full_model,
        reduce_op=reduction_op,
        random_init=random_init,
        training_mean=training_mean,
        training_std=training_std,
        is_dipole_moment=is_dipole_moment,
        is_spatial_extent=is_spatial_extent,
    )

    target_name = target_name if suffix is None else f"{target_name}" + suffix

    print(f"Number of parameters: {count_parameters(model)} \t Training {target_name}")
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    criterion = nn.MSELoss()

    iteration = 0
    # torch.autograd.set_detect_anomaly(True)
    epoch_loss = 0
    min_val_loss = 1e10

    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        epoch_loss = 0
        loss_running_mean = 0
        for batch in (pbar := tqdm(dataloader)):
            iteration += 1
            if self_supervised:
                coords, origin_coords, atoms, batch, labels = batch
                origin_coords = origin_coords.to(device).unsqueeze(2)
            else:
                coords, atoms, batch, labels = batch
            coords = coords.to(device)
            batch = batch.to(device)
            atoms = atoms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred, pred_coords = model(atoms, coords, batch)
            # loss = torch.sqrt(criterion(pred, labels))
            if self_supervised:
                loss = torch.sqrt(criterion(pred, labels)) + criterion(
                    pred_coords, origin_coords
                )
            else:
                loss = criterion(pred, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1e-1)
            optimizer.step()

            loss_running_mean = (
                loss.item()
                if loss_running_mean == 0
                else loss.item() * 0.01 + loss_running_mean * 0.99
            )
            pbar.set_postfix_str(f"{loss_running_mean:.4f}")
            epoch_loss = loss.item() * batch_size + epoch_loss

        scheduler.step()

        model.eval()
        val_loss = 0
        for batch in validation_loader:
            if self_supervised:
                coords, origin_coords, atoms, batch, labels = batch
                origin_coords = origin_coords.to(device).unsqueeze(2)
            else:
                coords, atoms, batch, labels = batch
            coords = coords.to(device)
            batch = batch.to(device)
            atoms = atoms.to(device)
            labels = labels.to(device)
            pred, pred_coords = model(atoms, coords, batch)
            if self_supervised:
                loss = torch.sqrt(criterion(pred, labels)) + criterion(
                    pred_coords, origin_coords
                )
            else:
                loss = criterion(pred, labels)
            val_loss = loss.item() * batch_size + val_loss

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.cpu().state_dict(), f"{target_name}_best_val_et_model.pt")

        print(
            f"Iteration [{iteration}]: Train Loss: {epoch_loss / len(dataset):.4f},",
            f"Validation Loss: {val_loss / len(validation_dataset):.4f}",
        )
    torch.save(model.cpu().state_dict(), f"{target_name}_et_model.pt")

    return model
