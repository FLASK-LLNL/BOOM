# MoLFormer 

## Installation

MoLFormer installation instructions are available at https://github.com/IBM/molformer. In our implementation, we perform optimization with the 'torch_optimizer' package and the Lamb algorithm instead of the original MoLFormer implementation which uses the 'apex' package with the FusedLAMB algorithm.


## Usage
First, download the MoLFormer checkpoint files from https://ibm.box.com/v/MoLFormer-data. These experiments used the N-Step-Checkpoint_0_0.ckpt and N-Step-Checkpoint_3_30000.ckpt files as the 'Scratch' and 'Pretrained' checkpoints, respectively. Copy these checkpoints into the data folder so that it looks like: 
```
data/
├── Pretrained MoLFormer
│   ├── checkpoints
│   │   ├── N-Step-Checkpoint_0_0.ckpt
│   │   └── N-Step-Checkpoint_3_30000.ckpt
```

Within each directory of the checkpoints folder, each model can be trained by running the corresponding 'run_finetune_{task_name}.sh' file. If you need to continue training from an existing checkpoint,
the 'run_finetune_{task_name}_continue.sh' file can be run instead. After model training, model predictions can be output by running the 'run_finetune_{task_name}_preds.sh' file.

## Results

### 10K CSD dataset- Scratch MoLFormer
<p align="center">
<img src="results/Pretrained_10k_dft_density_parity_plot.png" width="300" /> 
<img src="results/Pretrained_10k_dft_hof_parity_plot.png", width="300" />
</p>

### 10K CSD dataset- Pretrained MoLFormer
<p align="center">
<img src="results/Scratch_10k_dft_density_parity_plot.png" width="300" /> 
<img src="results/Scratch_10k_dft_hof_parity_plot.png", width="300" />
</p>

### QM9 dataset- Scratch MoLFormer
<p align="center">
<img src="results/Scratch_qm9_alpha_parity_plot.png" width="300" /> 
<img src="results/Scratch_qm9_cv_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_gap_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_homo_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_lumo_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_mu_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_r2_parity_plot.png", width="300" />
<img src="results/Scratch_qm9_zpve_parity_plot.png", width="300" />

</p>

### QM9 dataset- Pretrained MoLFormer
<p align="center">
<img src="results/Pretrained_qm9_alpha_parity_plot.png" width="300" /> 
<img src="results/Pretrained_qm9_cv_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_gap_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_homo_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_lumo_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_mu_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_r2_parity_plot.png", width="300" />
<img src="results/Pretrained_qm9_zpve_parity_plot.png", width="300" />

</p>
