# Pytorch implementation of a Variational Autoencoder with CNN encoder/decoder.
# Copyright (C) 2020  Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from torch import nn, sum
from torch.autograd import Variable

rng = np.random
np.set_printoptions(threshold=np.inf)


def init_weights(m):
    """ Initialize weights m """
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-0.08, 0.08)


class CNNVAE(nn.Module):
    """
        Pytorch class implements a convolution autoencoder (CAE) used for
        timeseries analysis.
    """
    def __init__(self, in_channels=1, sequence_length=1, latent_dim=16,
                 dev='cpu'):
        """
            Constructor of CNNAE class.

            Args:
                in_channels (int):      Number of input channels
                sequence_length (int):  Length of input sequence
                latent_dim (int):       Latent dimension

            Returns:
        """
        super(CNNVAE, self).__init__()
        self.in_channels = in_channels
        self.seq_len = sequence_length
        self.latent_dim = latent_dim
        self.dev = dev

        # Convolutions
        self.conv1 = nn.Conv1d(self.in_channels, 16, kernel_size=3,
                               stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(16, 4, kernel_size=3,
                               stride=1, padding=1, dilation=1)

        # Latent space
        self.m = nn.Linear(self.seq_len, self.latent_dim)
        self.s = nn.Linear(self.seq_len, self.latent_dim)
        self.l2h = nn.Linear(self.latent_dim, self.seq_len)

        # Deconvolutions
        self.conv2_t = nn.ConvTranspose1d(4, 16, kernel_size=2,
                                          stride=2, padding=0, dilation=1)
        self.conv1_t = nn.ConvTranspose1d(16, self.in_channels, kernel_size=2,
                                          stride=2, padding=0, dilation=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)

        self.m.apply(init_weights)
        self.s.apply(init_weights)

    def reparametrize(self, x):
        """
            Reparametrization tric method

            Args:
                x (tensor):     Hidden states (batch, seq_len x features)

            Returns:
                Mean (mu) and sigma (latent space)
        """
        if self.training is True:
            self.mu = self.m(x)
            log_sigma = self.s(x)
            self.std = log_sigma.exp_()
            eps = Variable(self.std.data.new(self.std.size()).normal_()
                           ).to(self.dev)
            return eps.mul(self.std).add_(self.mu)
        else:
            self.mu = self.m(x)
            return self.mu

    def var_loss(self, y_hat, y, loss_fn, beta=1):
        L = loss_fn(y_hat, y)
        scale = y.shape[2] * y.shape[1]
        KLD = -0.5 * sum(1 + self.std - self.mu.pow(2) - self.std.exp())
        KLD /= scale
        return L + beta * KLD

    def forward(self, x):
        """
            Forward method of CNNAE Class.

            Args:
                x (Torch Tensor): Input of shape (BSize, InChannels, SeqLen)

            Returns:
                output of size (BatchSize, InChannels, SeqLen)
        """
        out = self.relu(self.conv1(x))
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        m, n = out.shape[1], out.shape[2]
        out = out.view(-1, m * n)
        z = self.reparametrize(out)
        out = self.l2h(z)
        out = out.view(-1, m, n)
        out = self.relu(self.conv2_t(out))
        out = self.tanh(self.conv1_t(out))
        return out
