import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_len=120):
        super(VAE, self).__init__()

        self.conv1d1 = nn.Conv1d(input_len, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 11, kernel_size=10)

        self.fc1 = nn.Linear(110, 435)
        self.fc21 = nn.Linear(435, 292)
        self.fc22 = nn.Linear(435, 292)

        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)

    def encode(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = F.relu(self.fc1(h.view(-1, 110)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, 1, 292)
        h = z.repeat(1, 120, 1)
        out, h = self.gru(h)
        # TimeDistributed
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y = F.softmax(self.fc3(out_reshape), dim=1)
        y = y.contiguous().view(-1, 120, 35)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=True)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return BCE + KLD
