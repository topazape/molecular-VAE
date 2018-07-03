import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import VAE

def main():
    DEBUG = False
    torch.manual_seed(42)
    if DEBUG:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = np.load('./250k.npz')['arr']
    X_train, X_test = model_selection.train_test_split(X, random_state=42)

    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train.astype(np.float32))),
            shuffle=True, batch_size=250)

    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_test.astype(np.float32))),
            shuffle=False, batch_size=250)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    epochs = 10

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'epoch: {epoch}, loss: {loss}')
    else:
        torch.save(model.state_dict(), './epoch10.pth.tar')

if __name__ == '__main__':
    main()
