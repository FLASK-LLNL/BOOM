# Training MLP Models

Training MLP models doesn't require any additional package installation. Within the `MLP_train.py` file, simply update the following lines to specify your desired dataset:
```
###Example Usage for HoF- Update for your dataset of choice###
train_dataset = TrainHoFDataset()
iid_test_dataset = IDHoFDataset()
ood_test_dataset = OODHoFDataset()
target='hof'
######
```
The `target` string can technically be anything, it only specifies the name of the folder where the model outputs will be stored.
