# Random Forest + RDKit feautres

## Installation

The random forest model requires `RDKit` and `deepchem` to be installed. **NOTE: This code currently only works with `deepchem==2.6.1`

Install `deepchem` with pip:

```bash
pip install deepchem==2.6.1
```

Also, `flask_ood` must also be installed of course. 

## Usage
Run all the experiments using the following command:

```bash
python run_experiments.py
```

## Results

### 10K CSD dataset

<p align="center">
<img src="results/Density/Density_parity_plot.png" width="300" /> 
<img src="results/HoF/HoF_parity_plot.png", width="300" />

### QM9 Datasets
<p align="center">
<img src="results/alpha/alpha_parity_plot.png" width="300" /> 
<img src="results/cv/cv_parity_plot.png" width="300" /> 
<img src="results/gap/gap_parity_plot.png" width="300" /> 
<img src="results/homo/homo_parity_plot.png" width="300" /> 
<img src="results/lumo/lumo_parity_plot.png" width="300" /> 
<img src="results/mu/mu_parity_plot.png" width="300" /> 
<img src="results/r2/r2_parity_plot.png" width="300" /> 
<img src="results/zpve/zpve_parity_plot.png" width="300" /> 

</p>
