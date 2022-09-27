# Pytorch implementation of an Autoencoder with Causal CNN encoder/decoder.
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
from torch import nn


rng = np.random


def CausalConv1d(in_channels, out_channels, kernel_size=2, dilation=1,
                 **kwargs):
    """
        Implements a Causal 1D Convolution.

        Args:
            in_channels (int):  Number of input channels
            out_channels (int): Number of ouput channels
            kernel_size (int):  Kernel size
            Dilation (int):     Dilation factor

        Returns:
            Convolution method

        Notes:
        After calling this function use the following chorp of the output
        x = x[:, :, :-self.conv1.padding[0]]
    """
    pad = (kernel_size - 1) * dilation
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=pad, dilation=dilation, **kwargs)
    return conv


class CausalCNNAE(nn.Module):
    """
        Pytorch class implements a causal convolution autoencoder (CAE) used
        for timeseries analysis.
    """
    def __init__(self, in_channels=1, sequence_length=1, latent_dim=16):
        """
            Constructor of CausalCNNAE class.

            Args:
                in_channels (int):      Number of input channels
                sequence_length (int):  Length of input sequence
                latent_dim (int):       Latent dimension

            Returns:
        """
        super(CausalCNNAE, self).__init__()
        self.in_channels = in_channels
        self.seq_len = sequence_length

        # Convolutions
        self.conv1 = CausalConv1d(self.in_channels, 16, kernel_size=3,
                                  dilation=1, stride=1)
        self.conv2 = CausalConv1d(16, 4, kernel_size=3, dilation=2, stride=1)

        # Latent space
        self.h2l = nn.Linear(self.seq_len, latent_dim)
        self.l2h = nn.Linear(latent_dim, self.seq_len)

        # Deconvolutions
        self.conv2_t = nn.ConvTranspose1d(4, 16, kernel_size=2,
                                          stride=2, padding=0, dilation=1)
        self.conv1_t = nn.ConvTranspose1d(16, self.in_channels, kernel_size=2,
                                          stride=2, padding=0, dilation=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
            Forward method of CNNAE Class.

            Args:
                x (Torch Tensor): Input of shape (BSize, InChannels, SeqLen)

            Returns:
                output of size (BatchSize, InChannels, SeqLen)
        """
        out = self.relu(self.conv1(x))
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.pool(out)
        m, n = out.shape[1], out.shape[2]
        out = out.view(-1, m * n)
        out = self.h2l(out)
        out = self.l2h(out)
        out = out.view(-1, m, n)
        out = self.relu(self.conv2_t(out))
        out = self.tanh(self.conv1_t(out))
        return out
