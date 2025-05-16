# MACE
## Installation

Apart from the `boom` requirements, the model requires the following packages:
```
torch
torch-geometric
mace
ase
rdkit
```

## Usage
Create the property splits using the following command:
```bash
python make_splits.py
```

Train for a given property using the following command:
```bash
mace_run_train --config=configs/{property}.yaml
```
where `{property}` is one of `[density, hof, alpha, cv, gap, homo, lumo, mu, r2, zpve]`

After training for all of the properties, create plots using the following command:
```bash
python make_plots.py
```


## Results

### 10K CSD dataset

<p align="center">
<img src="results/mace_density.png" width="300" /> 
<img src="results/mace_hof.png", width="300" />

### QM9 dataset
<p align="center">
<img src="results/mace_alpha.png" width="300" /> 
<img src="results/mace_cv.png", width="300" />
<img src="results/mace_gap.png", width="300" />
<img src="results/mace_homo.png", width="300" />
<img src="results/mace_lumo.png", width="300" />
<img src="results/mace_mu.png", width="300" />
<img src="results/mace_r2.png", width="300" />
<img src="results/mace_zpve.png", width="300" />
</p>
