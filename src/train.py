from itertools import chain
import torch
import torch.utils.data
from torch import optim
import h5py
from models import MolEncoder, MolDecoder

h5f = h5py.File('../data/zinc12.h5')
data_train = h5f['data_train'][:]
data_test = h5f['data_test'][:]
charset = h5f['charset'][:]
h5f.close()

data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)

train = torch.utils.data.TensorDataset(data_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=250, shuffle=True)

encoder = MolEncoder(c=len(charset))
decoder = MolDecoder(c=len(charset))
encoder.to('cuda')
decoder.to('cuda')
optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()))

epochs = 1

for epoch in range(epochs):
    encoder.train()
    decoder.train()

    for batch_idx, x in enumerate(train_loader):
        x_var = x[0].to('cuda')
        y_var = encoder(x_var)
        z_var = decoder(y_var)
        loss = encoder.vae_loss(z_var, x_var)
        if batch_idx % 100 == 0:
            print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(encoder.state_dict(), f'./models/encoder-{epoch}.pth')
    torch.save(decoder.state_dict(), f'./models/decoder-{epoch}.pth')
