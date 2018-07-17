import torch
import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE

model = MolecularVAE()
model.load_state_dict(torch.load('./ref/vae-017-43.437255859375.pth'))
model.to('cuda')
model.eval()

start = 'C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]'
start = start.ljust(120)
oh = OneHotFeaturizer()
start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to('cuda')

recon_x = model(start_vec)[0].cpu().detach().numpy()
y = np.argmax(recon_x, axis=2)
print(start)
print(oh.decode_smiles_from_index(y[0]))
