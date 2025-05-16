# Graphormer (3D)

## Additional dependencies

- This graphormer implementation treats molecular data as graphs (following PyG format), not sequences.  
- PyTorch Lightning was used to train and save the models. Technically you don't need Lightning to retrieve the model checkpoints, but it's fairly straightforward to install Lightning with `pip`.

```
torch_geometric
torch_scatter
torch_cluster
lightning
```

Once PyTorch and RDKit are installed, here is what I did:
```
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install lightning
```

Notice the above code for installing `torch_scatter` and `torch_cluster`. The PyTorch and CUDA versions need to match the ones on your machine.

## How to use?

Run `demo.ipynb` to *interactively* load/train the model, make predictions, and generate parity plots.

Run `python run_experiment.py` to generate the results (namely the parity plots).

- The `results` folder contains the parity plots
- The `src` folder contains the source code
- The `lightning_logs` folder contains the model checkpoint files
- The `cached-3d-data` folder contains the cached 3D files (for the 10K dataset)