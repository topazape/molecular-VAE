import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from .utils import Flatten, Repeat, TimeDistributed

__all__ = ['MolEncoder', 'MolDecoder']


class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             SELU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def __init__(self, i=435, o=292, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class MolEncoder(nn.Module):

    def __init__(self, i=120, o=292, c=35):
        super(MolEncoder, self).__init__()

        self.i = i

        #self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_1 = nn.Conv1d(i, 9, kernel_size=9)
        #self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        #self.conv_3 = ConvSELU(9, 10, kernel_size=11)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        #self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435), SELU(inplace=True))
        self.dense_1 = nn.Linear((c - 29 + 3) * 10, 435)

        #self.lmbd = Lambda(435, o)
        self.lmbd = Lambda(435, 2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = F.relu(self.conv_3(out))
        #out = Flatten()(out)
        out = out.view(out.size(0), -1)
        out = F.selu(self.dense_1(out))

        return self.lmbd(out)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        #bce = nn.BCELoss(size_average=True)
        #xent_loss = self.i * bce(x_decoded_mean, x.detach())
        xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=2, o=120, c=35):
        super(MolDecoder, self).__init__()

        #self.latent_input = nn.Sequential(nn.Linear(i, i), SELU(inplace=True))
        self.latent_input = nn.Linear(i, i)
        #self.repeat_vector = Repeat(o)
        self.gru = nn.GRU(i, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, c)
        #self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
        #                                                  nn.Softmax())
        #                                    )

    def forward(self, x):
        out = F.selu(self.latent_input(x))
        #out = self.repeat_vector(out)
        out = out.view(out.size(0), 1, out.size(-1)).repeat(1, 120, 1)
        out, h = self.gru(out)

        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape))
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        #return self.decoded_mean(out)
        return y
