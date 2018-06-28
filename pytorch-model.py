import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = np.load('./5k.npz')['arr']
X_train, X_test = model_selection.train_test_split(X, random_state=42)

train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train.astype(np.float32))),
        shuffle=True, batch_size=128)

test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test.astype(np.float32))),
        shuffle=True, batch_size=128)


class VAE(nn.Module):
    def __init__(self, input_len=120):
        super(VAE, self).__init__()

        self.conv1d1 = nn.Conv1d(input_len, out_channels=9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)

        self.fc1 = nn.Linear(90, 435)
        self.fc21 = nn.Linear(435, 292)
        self.fc22 = nn.Linear(435, 292)

        self.gru1 = nn.GRU(501, 200)
        self.fc5 = nn.Linear(200, 400)
        self.fc6 = nn.Linear(400, 784)

    def encode(self, x):
        print(x.size())
        h = F.relu(self.conv1d1(x))
        print(h.size())
        h = F.relu(self.conv1d2(h))
        print(h.size())
        h = F.relu(self.conv1d3(h))
        print(h.size())
        h = F.relu(self.fc1(h.view(-1, 90)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return F.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    for epoch in range(1, 51):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
