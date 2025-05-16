## ModernBERT

### Additional libraries
The ModernBERT model is implemented using the [Transformers](https://huggingface.co/docs/transformers/en/index)
library. To install the required libraries, run the following command:

```bash
pip install transformers datasets
```
Also, `PyTorch` must be installed as well. 


### Training the model
We provide a training script to train a ModernBERT model on the BOOM dataset. The 
script can be run with the following command:

```bash
python train.py prop1,prop2,...
```

where `prop1,prop2,...` are the properties to train the model on. The properties are:
- `HoF`
- `Density`
- `Alpha`
- `Mu`
- `R2`
- `Cv`
- `HOMO`
- `LUMO`
- `Gap`
