# ChemBerta 
  
## Installation

Environment can be created from the provided `environment.yml` file

```bash
conda env create -f environment.yml
```


Also, `boom` must also be installed of course.

## Usage
Run all the experiments by running each of the `run_experiment_{prop_name}.py` files. Each python script will train both a pre-trained ChemBERTa model and a scratch ChemBERTa model.

## Results

### 10K CSD dataset

<p align="center">
<img src="results/Density_pre_trained=True_parity_plot.png", width="300" />
<img src="results/Density_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/HoF_pre_trained=True_parity_plot.png", width="300" />
<img src="results/HoF_pre_trained=False_parity_plot.png", width="300" />
</p>

### QM9 dataset
<p align="center">
<img src="results/QM9_alpha_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_alpha_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_cv_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_cv_pre_trained=False_parity_plot.png", width="300" />
</p>


<p align="center">
<img src="results/QM9_gap_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_gap_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_homo_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_homo_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_lumo_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_lumo_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_mu_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_mu_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_r2_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_r2_pre_trained=False_parity_plot.png", width="300" />
</p>

<p align="center">
<img src="results/QM9_zpve_pre_trained=True_parity_plot.png", width="300" />
<img src="results/QM9_zpve_pre_trained=False_parity_plot.png", width="300" />
</p>
