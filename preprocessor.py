import numpy as np
from featurizer import OneHotFeaturizer

with open('./5k.smi') as f:
    smiles = []
    for smi in f:
        smiles.append(smi.rstrip())

ohf = OneHotFeaturizer()
oh_smiles = ohf.featurize(smiles)
np.savez_compressed('5k.npz', arr=oh_smiles)
