import torch
import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE

model = MolecularVAE()
model.load_state_dict(torch.load('./models/vae-50.pth'))
model.to('cuda')
model.eval()

start = 'c1ccccn1'
start = start.ljust(120)
oh = OneHotFeaturizer()
start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to('cuda')

recon_x = model(start_vec)[0].cpu().detach().numpy()
y = np.argmax(recon_x, axis=2)
print(oh.decode_smiles_from_index(y[0]))
