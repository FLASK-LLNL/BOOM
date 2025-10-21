import sklearn
from sklearn.neural_network import MLPRegressor
import time
import numpy as np
import os
import pickle
import deepchem as dc
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from boom.datasets.SMILESDataset import *


def get_rdkit_features(mols):
    featurizer = dc.feat.RDKitDescriptors()
    features = featurizer.featurize(mols)
    feature_names = np.array([f[0] for f in Descriptors.descList]) #weird error here with feature_names, sometimes doesn't work??

    ind, features = zip(*[(i, feat) for i, feat in enumerate(features) if len(feat) != 0])
    ind = list(ind)
    # breakpoint()
    features = np.array(features)


    mols = mols[ind]

    #density_data = density_data.iloc[ind]
    #density_refcodes = density_data['refcode'].to_numpy()
    #property_output = property_data
    # log Ipc transformation
    ipc_idx = np.where(feature_names == 'Ipc')
    feature_names[ipc_idx] = 'log_Ipc'
    features[:, ipc_idx] = np.log(features[:, ipc_idx] + 1)

    # remove constants:
    #nonzero_sd = np.where(~np.isclose(np.std(features, axis=0), 0))[0]
    #features = features[:, nonzero_sd]
    #feature_names = feature_names[nonzero_sd]

    #remove nan's (for polymers)
    #good_indices=np.array(range(len(features[0])))
    #for i in range(len(features)):
    #    for j in range(len(features[0])):
    #        if(str(features[i][j])=='nan' and j in good_indices):
    #            good_indices = np.delete(good_indices, np.where(good_indices == j))

    #features = features[:, good_indices]
    #feature_names = feature_names[good_indices]

    # cor == 1 with other features:
    corr_features = np.array(['Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
        'MaxAbsEStateIndex', 'ExactMolWt', 'NumHeteroatoms'])
    #features = features[:, ~np.isin(feature_names, corr_features)]
    feature_names = feature_names[~np.isin(feature_names, corr_features)]
    print(np.shape(features))
    return(features,feature_names)

def dataset_wrapper_lipo(dataset, normalizing_dataset):
    num_samples = len(dataset)
    mols = np.array([Chem.MolFromSmiles(smiles) for smiles, _ in dataset])
    features, feature_names = get_rdkit_features(mols)
    labels=np.array(dataset[:,1],dtype=float)
    #labels = np.array([target for _, target in dataset])
    mean = np.mean(np.array(normalizing_dataset[:,1],dtype=float))
    std = np.std(np.array(normalizing_dataset[:,1],dtype=float))
    labels = (labels - mean) / std
    return features, labels

def denormalize_target_lipo(target, normalizing_dataset):
    mean = np.mean(np.array(normalizing_dataset[:,1],dtype=float))
    std = np.std(np.array(normalizing_dataset[:,1],dtype=float))
    return target * std + mean

###Example Usage for HoF- Update for your dataset of choice###
train_dataset = TrainHoFDataset()
iid_test_dataset = IDHoFDataset()
ood_test_dataset = OODHoFDataset()
target='hof'
######

train_dataset=np.column_stack((train_dataset.smiles, train_dataset.property_values))
ood_test_dataset=np.column_stack((ood_test_dataset.smiles, ood_test_dataset.property_values))
iid_test_dataset=np.column_stack((iid_test_dataset.smiles, iid_test_dataset.property_values))
print('Making features')
train_features, train_y = dataset_wrapper_lipo(train_dataset, train_dataset)
train_y = denormalize_target_lipo(train_y, train_dataset)
iid_features, iid_y = dataset_wrapper_lipo(iid_test_dataset, train_dataset)
ood_features, ood_y = dataset_wrapper_lipo(ood_test_dataset, train_dataset)
iid_real_vals = denormalize_target_lipo(iid_y, train_dataset)
ood_real_vals = denormalize_target_lipo(ood_y, train_dataset)

#get rid of any nans-train
good_samples=[]
for i in range(len(train_features)):
    if(np.isnan(train_features[i]).any()):
        continue
    else:
        good_samples.append(i)
train_features=train_features[good_samples]
train_y=train_y[good_samples]

good_samples=[]
for i in range(len(iid_features)):
    if(np.isnan(iid_features[i]).any()):
        continue
    else:
        good_samples.append(i)
iid_features=iid_features[good_samples]
iid_y=iid_y[good_samples]

good_samples=[]
for i in range(len(ood_features)):
    if(np.isnan(ood_features[i]).any()):
        continue
    else:
        good_samples.append(i)
ood_features=ood_features[good_samples]
ood_y=ood_y[good_samples]

for seed in range(3):
    regr = MLPRegressor(random_state=seed, max_iter=2000, tol=0.1,hidden_layer_sizes=(300,300))
    print('fitting')
    start=time.time()
    regr.fit(train_features, train_y)
    end=time.time()
    print(str(end-start) + ' seconds taken for training')
    print('done model fitting')
    train_y_pred = regr.predict(train_features)
    train_y_pred = denormalize_target_lipo(train_y_pred, train_dataset[1:])

    mean_score = np.mean(np.abs(train_y - train_y_pred))
    std_score = np.std(np.abs(train_y - train_y_pred))

    iid_preds = regr.predict(iid_features)
    ood_preds = regr.predict(ood_features)

    iid_preds = denormalize_target_lipo(iid_preds, train_dataset)
    ood_preds = denormalize_target_lipo(ood_preds, train_dataset)


    iid_smiles = [smiles for smiles, _ in iid_test_dataset]
    ood_smiles = [smiles for smiles, _ in ood_test_dataset]
    results = {
        "target": target,  # "Density", "HoF", or "gap"
        "mean_score": mean_score,
        "std_score": std_score,
        "iid_smiles": iid_smiles,
        "pred_iid_vals": iid_preds,
        "real_iid_vals": iid_real_vals,
        "ood_smiles": ood_smiles,
        "pred_ood_vals": ood_preds,
        "real_ood_vals": ood_real_vals,
    }

    target = results["target"]
    pred_iid_vals = np.array(results["pred_iid_vals"])
    real_iid_vals = np.array(results["real_iid_vals"])
    pred_ood_vals = np.array(results["pred_ood_vals"])
    real_ood_vals = np.array(results["real_ood_vals"])
    os.makedirs(f'./{target}',exist_ok=True)

    np.save(f"./{target}/{target}{seed}_iid_preds.npy", pred_iid_vals)
    np.save(f"./{target}/{target}{seed}_iid_real.npy", real_iid_vals)
    np.save(f"./{target}/{target}{seed}_ood_preds.npy", pred_ood_vals)
    np.save(f"./{target}/{target}{seed}_ood_real.npy", real_ood_vals)

