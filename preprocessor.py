import numpy as np
from featurizer import OneHotFeaturizer

with open('./250k_rndm_zinc_drugs_clean.smi') as f:
    smiles = []
    for smi in f:
        smiles.append(smi.rstrip())

ohf = OneHotFeaturizer()
oh_smiles = ohf.featurize(smiles)
np.savez_compressed('250k.npz', arr=oh_smiles)
