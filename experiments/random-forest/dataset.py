################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import numpy as np
from rdkit import Chem
import deepchem as dc
from rdkit.Chem import Descriptors

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
    good_indices=np.array(range(len(features[0])))
    for i in range(len(features)):
        for j in range(len(features[0])):
            if(str(features[i][j])=='nan' and j in good_indices):
                good_indices = np.delete(good_indices, np.where(good_indices == j))
    
    features = features[:, good_indices]
    feature_names = feature_names[good_indices]
    
    # cor == 1 with other features:
    corr_features = np.array(['Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 
        'MaxAbsEStateIndex', 'ExactMolWt', 'NumHeteroatoms'])
    features = features[:, ~np.isin(feature_names, corr_features)]
    feature_names = feature_names[~np.isin(feature_names, corr_features)]
    print(np.shape(features))
    return(features,feature_names)


def dataset_wrapper(dataset, normalizing_dataset):
    num_samples = len(dataset)
    mols = np.array([Chem.MolFromSmiles(smiles) for smiles, _ in dataset])
    features, feature_names = get_rdkit_features(mols)
    labels = np.array([target for _, target in dataset])
    mean = np.mean([target for _, target in normalizing_dataset])
    std = np.std([target for _, target in normalizing_dataset])
    labels = (labels - mean) / std
    return features, labels

def denormalize_target(target, normalizing_dataset):
    mean = np.mean([target for _, target in normalizing_dataset])
    std = np.std([target for _, target in normalizing_dataset])
    return target * std + mean
