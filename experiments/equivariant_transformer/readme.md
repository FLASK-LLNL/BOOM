# Equivariant Transformer
## Installation

Apart from the `flask_ood` requirements, the model requires the following packages:
```
torch
torch-geometric
torchmd-net
rdkit
```

**Note**: `torch` and  the cuda toolkits are available on conda and better to install from that. `rdkit` is easier to install via `pip`. In my experience `torchmd-net` is easier to install from source. 

## Pre-trained models

Get the pre-trained model from Torch-MD example
```
wget http://pub.htmd.org/et-qm9.zip
unzip et-qm9.zip
mv epoch=649-val_loss=0.0003-test_loss=0.0059.ckpt pre-trained.ckpt
```

**Noet**: The download may take a few minutes. I may upload the model to a more permanent location in the future.

## Usage
Run all the experiments using the following command:

```bash
python run_experiments.py
```


## Results

### 10K CSD dataset

<p align="center">
<img src="results/Density_parity_plot.png" width="300" /> 
<img src="results/HoF_parity_plot.png", width="300" />
</p>