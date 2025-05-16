################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch
import numpy as np

####################### Lightning modules #######################

from src.datamodules import HighEnergyDataModule_ood_3d
from src.modules import LitGraphormer_3d

den_datamodule = HighEnergyDataModule_ood_3d(
    data_fname   = '../../flask_ood/data/10k_dft_data_with_ood_splits.csv',
    prop         = 'den',
    path_3d_file = './cached-3d-data/3d-rdkit.pkl',
    batch_size   = 64,
    num_workers  = 4,
)

den_graphormer = LitGraphormer_3d.load_from_checkpoint(
    './lightning_logs/graphormer-10k-den/ood-3d-rdkit/version_0/checkpoints/epoch=2919-step=400000.ckpt',
    sum_pooling = False,
)

hof_datamodule = HighEnergyDataModule_ood_3d(
    data_fname   = '../../flask_ood/data/10k_dft_data_with_ood_splits.csv',
    prop         = 'hof',
    path_3d_file = './cached-3d-data/3d-rdkit.pkl',
    batch_size   = 64,
    num_workers  = 4,
)

hof_graphormer = LitGraphormer_3d.load_from_checkpoint(
    './lightning_logs/graphormer-10k-hof/ood-3d-rdkit/version_0/checkpoints/epoch=2898-step=400000.ckpt',
    sum_pooling = True,
)

####################### Inference #######################

from tqdm import tqdm

def predict(model, loader, y_scaler=None):
    pred_y, true_y = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            pred_y.append(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
            true_y.append(batch.y)
    pred_y = torch.vstack(pred_y).numpy()
    true_y = torch.vstack(true_y).numpy()

    if y_scaler is not None:
        pred_y = y_scaler.inverse_transform(pred_y)
        true_y = y_scaler.inverse_transform(true_y)
    
    return pred_y.reshape(-1), true_y.reshape(-1)

####################### Make parity plots #######################

import matplotlib.pyplot as plt; plt.rc("font", weight="bold")
import seaborn as sns

################ Density ################

den_graphormer.eval(); den_graphormer.to('cpu')
den_datamodule.setup()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Model predictions
id_pred_y, id_true_y   = predict(den_graphormer.ema_model, den_datamodule.val_dataloader(),  den_datamodule.y_scaler)
ood_pred_y, ood_true_y = predict(den_graphormer.ema_model, den_datamodule.test_dataloader(), den_datamodule.y_scaler)

# Measure RMSE
id_rmse  = np.sqrt(np.mean((id_pred_y - id_true_y)**2))
ood_rmse = np.sqrt(np.mean((ood_pred_y - ood_true_y)**2))

# Ideal line
extent = [1, 2]
ax.plot(extent, extent, "r--", linewidth=3.5, alpha=1, label="Perfect Prediction", zorder=0)

# ID and OOD performance
sns.scatterplot(x=ood_true_y, y=ood_pred_y, ax=ax, alpha=1, size=5, color='darkblue',  legend=False, label=f'OOD ({ood_rmse:.4f})')
sns.scatterplot(x=id_true_y,  y=id_pred_y,  ax=ax, alpha=1, size=5, color='darkorange',legend=False, label=f'ID ({id_rmse:.4f})')

ax.set_title(f'Graphormer 3D - Density', weight='bold', fontsize='x-large')
ax.set_xlabel(f"True Density",      weight="bold", fontsize="x-large")
ax.set_ylabel(f"Predicted Density", weight="bold", fontsize="x-large")
ax.xaxis.set_tick_params(labelsize='large')
ax.yaxis.set_tick_params(labelsize='large')
ax.legend(prop={'weight': 'bold'})

plt.tight_layout()
plt.savefig(f'./results/Density-parity-plot.png', dpi=150)

################ HoF ################

hof_graphormer.eval(); hof_graphormer.to('cpu')
hof_datamodule.setup()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Model predictions
id_pred_y, id_true_y   = predict(hof_graphormer.ema_model, hof_datamodule.val_dataloader(),  hof_datamodule.y_scaler)
ood_pred_y, ood_true_y = predict(hof_graphormer.ema_model, hof_datamodule.test_dataloader(), hof_datamodule.y_scaler)

# Measure RMSE
id_rmse  = np.sqrt(np.mean((id_pred_y - id_true_y)**2))
ood_rmse = np.sqrt(np.mean((ood_pred_y - ood_true_y)**2))

# Ideal line
extent = [-1000, 400]
ax.plot(extent, extent, "r--", linewidth=3.5, alpha=1, label="Perfect Prediction", zorder=0)

# ID and OOD performance
sns.scatterplot(x=ood_true_y, y=ood_pred_y, ax=ax, alpha=1, size=5, color='darkblue',  legend=False, label=f'OOD ({ood_rmse:.4f})')
sns.scatterplot(x=id_true_y,  y=id_pred_y,  ax=ax, alpha=1, size=5, color='darkorange',legend=False, label=f'ID ({id_rmse:.4f})')

ax.set_title(f'Graphormer 3D - HoF', weight='bold', fontsize='x-large')
ax.set_xlabel(f"True HoF",      weight="bold", fontsize="x-large")
ax.set_ylabel(f"Predicted HoF", weight="bold", fontsize="x-large")
ax.xaxis.set_tick_params(labelsize='large')
ax.yaxis.set_tick_params(labelsize='large')
ax.legend(prop={'weight': 'bold'})

plt.tight_layout()
plt.savefig(f'./results/HoF-parity-plot.png', dpi=150)
