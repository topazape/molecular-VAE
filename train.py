import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import h5py
from models import MolecularVAE


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

X = np.load('./data/250k.npz')['arr'].astype(np.float32)

train = torch.utils.data.TensorDataset(torch.from_numpy(X))
train_loader = torch.utils.data.DataLoader(train, batch_size=250, shuffle=True)
torch.manual_seed(42)

epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MolecularVAE().to(device)
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].transpose(1,2).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.transpose(1,2), mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'{epoch} / {batch_idx}\t{loss:.4f}')
    print('train', train_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        data = data[0].transpose(1,2).to(device)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data.transpose(1,2), mu, logvar).item()
    print('test', test_loss / len(test_loader))

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    if epoch % 1 == 0:
        torch.save(model.state_dict(),
                './ref/vae-{:03d}-{}.pth'.format(epoch, train_loss))
