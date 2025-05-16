# GNN Representation
## Installation

Apart from `flask_ood` requirements, the model requires the following packages:
```
torch
rdkit
```

**Note**: `torch` and  the cuda toolkits are available on conda and better to install from that. `rdkit` is easier to install via `pip`.

## Usage

Run all the experiments using the following command:

```bash
python run_experiments.py
```

## Results

### 10K CSD dataset

#### Topological GNN
<p align="center">
<img src="results/TopologicalGNN_Density_parity_plot.png" width="300" /> 
<img src="results/TopologicalGNN_HoF_parity_plot.png", width="300" />
</p>

#### 3D Invariant GNN
<p align="center">
<img src="results/InvariantGNN_Density_parity_plot.png" width="300" /> 
<img src="results/InvariantGNN_HoF_parity_plot.png", width="300" />
</p>

#### 3D Equivariant GNN
<p align="center">
<img src="results/EquivariantGNN_Density_parity_plot.png" width="300" /> 
<img src="results/EquivariantGNN_Density_parity_plot.png", width="300" />
</p>


### 10K CSD dataset
---
#### Topological GNN
<p align="center">
<img src="results/TopologicalGNN_Gap_parity_plot.png" width="300" /> 
<img src="results/TopologicalGNN_HOMO_parity_plot.png", width="300" />
<img src="results/TopologicalGNN_LUMO_parity_plot.png" width="300" /> 
<img src="results/TopologicalGNN_Alpha_parity_plot.png", width="300" />
<img src="results/TopologicalGNN_CV_parity_plot.png", width="300" />
<img src="results/TopologicalGNN_R2_parity_plot.png", width="300" />
<img src="results/TopologicalGNN_ZPVE_parity_plot.png", width="300" />
<img src="results/TopologicalGNN_Mu_parity_plot.png", width="300" />
</p>