import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import h5py
from models import VAE


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.GRU):
        for weights in m.all_weights:
            for weight in weights:
                if len(weight.size()) > 1:
                    nn.init.xavier_uniform_(weight.data)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#h5f = h5py.File('./data/zinc12.h5')
#X = h5f['data_train'][:]
#X = X.transpose(0, 2, 1)
X = np.load('./data/250k-T.npz')['arr'].astype(np.float32)
#X_train, X_test = model_selection.train_test_split(X, random_state=1022)

#train = torch.utils.data.TensorDataset(torch.from_numpy(X_train))
#train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

train = torch.utils.data.TensorDataset(torch.from_numpy(X))
train_loader = torch.utils.data.DataLoader(train, batch_size=250, shuffle=False)

#test = torch.utils.data.TensorDataset(torch.from_numpy(X_test))
#test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

#torch.manual_seed(42)
epochs = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VAE().to(device)
model.apply(initialize_weights)
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'{epoch} / {batch_idx}\t{loss:.4f}')
    print('train', train_loss / len(train_loader.dataset))

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        data = data[0].to(device)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()
    print('test', test_loss / len(test_loader))

for epoch in range(1, epochs + 1):
    train(epoch)
    #test(epoch)
    if epoch % 1 == 0:
        torch.save(model.state_dict(), './models/vae-{}.pth'.format(epoch))
