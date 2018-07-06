import numpy as np
from featurizer import OneHotFeaturizer

with open('./data/250k_rndm_zinc_drugs_clean.smi') as f:
    smiles = []
    for smi in f:
        smiles.append(smi.rstrip())

ohf = OneHotFeaturizer()
oh_smiles = ohf.featurize(smiles)
print(oh_smiles.shape)
np.savez_compressed('./data/250k-T.npz', arr=oh_smiles)
