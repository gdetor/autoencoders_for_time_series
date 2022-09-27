# Pytorch implementation of vanilla Variational Autoencoder with feed-forward
# encoder/decoder modules.
# Copyright (C) 2020 Georgios Is. Detorakis
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
from torch import nn
from torch import sum
from torch.autograd import Variable


def init_weights(m):
    """
    Initialize weights for linear layers.
    """
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-0.08, 0.08)


class LENC(nn.Module):
    """
        Pytorch class implements a linear encoder (LENC) used for
        timeseries analysis.
    """
    def __init__(self, shape=(100, 10), funs=(nn.ReLU(inplace=True))):
        """
            Constructor of LENC class.

            Args:
                shape (tuple ints):     Number of neurons per layer and number
                                        of layers (tuple's length)
                funs (tuple funcs):     A tuple populated with all the
                                        non-lineaer functions for each layer

            Returns:
        """
        super(LENC, self).__init__()
        self.nlayers = len(shape)       # Number of layers
        self.layers = []                # List of layers
        # Initialize the Linear layers and add Batch Normalization
        for i in range(self.nlayers-1):
            in_dim = shape[i]
            out_dim = shape[i+1]
            self.layers.append(nn.BatchNorm1d(in_dim))
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(funs[i])

        self.net_encoder = nn.Sequential(*self.layers)
        self.net_encoder.apply(init_weights)

    def forward(self, x):
        """
            Forward method of LENC Class.

            Args:
                x (Torch Tensor): Input of shape (BSize, SeqLen, #features)

            Returns:
                output of size (BatchSize, SeqLen, #features)
        """
        return self.net_encoder(x)


class LDEC(nn.Module):
    """
        Pytorch class implements a linear decoder (LDEC) used for
        timeseries analysis.
    """
    def __init__(self, shape=(10, 100), funs=(nn.ReLU(inplace=True)),
                 vae=False, ldim=16):
        """
            Constructor of LDEC class.

            Args:
                shape (tuple ints):     Number of neurons per layer and number
                                        of layers (tuple's length)
                funs (tuple funcs):     A tuple populated with all the
                                        non-lineaer functions for each layer
                vae (bool):             Determines if the variational AE is
                                        enabled
                ldim (int):             Latent dimension

            Returns:
        """
        super(LDEC, self).__init__()
        self.nlayers = len(shape)
        self.layers = []

        # In case of variational AE we need to initialize all the necessary
        # linear layers for mu, sigma, and z
        if vae is True:
            self.layers.append(nn.BatchNorm1d(ldim))
            self.layers.append(nn.Linear(ldim, shape[0]))
            self.layers.append(funs[0])

        # Initialize all the linear layers of the decoder
        for i in range(self.nlayers-1):
            in_dim = shape[i]
            out_dim = shape[i+1]
            self.layers.append(nn.BatchNorm1d(in_dim))
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(funs[i+1])
        # self.layers[-1] = funs[-1]
        self.net_decoder = nn.Sequential(*self.layers)
        self.net_decoder.apply(init_weights)

    def forward(self, x):
        """
            Forward method of LDEC Class.

            Args:
                x (Torch Tensor): Input of shape (BSize, SeqLen, #features)

            Returns:
                output of size (BatchSize, SeqLen, #features)
        """
        return self.net_decoder(x)


class LVAE(nn.Module):
    """
        Pytorch class implements a linear (V)AE used for timeseries analysis.
    """
    def __init__(self, ldim=16, shape=(100, 10), device="cpu", vae=True,
                 funs_enc=(nn.ReLU(inplace=True),),
                 funs_dec=(nn.ReLU(inplace=True),)):
        """
            Constructor of LVAE class.

            Args:
                shape (tuple ints):     Number of neurons per layer and number
                                        of layers (tuple's length)
                funs_enc (tuple funcs):   A tuple populated with all the
                                          non-lineaer functions for each layer
                                          for the encoder
                funs_dec (tuple funcs):   A tuple populated with all the
                                          non-lineaer functions for each layer
                                          for the decoder
                vae (bool):             Determines if the variational AE is
                                        enabled
                ldim (int):             Latent dimension
                devive (torch dev):     Determines the device on which the
                                        model will be running

            Returns:
        """
        super(LVAE, self).__init__()
        self.shape_dec = shape[::-1]
        self.dev = device

        self.encoder = LENC(shape=shape, funs=funs_enc)
        self.decoder = LDEC(shape=shape[::-1], funs=funs_dec, vae=vae,
                            ldim=ldim)

        self.m = nn.Linear(shape[-1], ldim)
        self.s = nn.Linear(shape[-1], ldim)

        self.m.apply(init_weights)
        self.s.apply(init_weights)

    def reparametrize(self, x):
        """
            Reparametrization method. Applies the reparametrization trick on
            the latent space.

            Args:
                x (Torch Tensor): Input of shape (BSize, SeqLen, #features)

            Returns:
                output of size (BatchSize, SeqLen, #features)
        """
        if self.training is True:
            self.mu = self.m(x)
            self.log_sigma = self.s(x)
            self.std = self.log_sigma.exp_()
            eps = Variable(self.std.data.new(self.std.size()).normal_()
                           ).to(self.dev)
            return eps.mul(self.std).add_(self.mu)
        else:
            self.mu = self.m(x)
            return self.mu

    # def var_loss(self, y_hat, y, batch_size=32, inp_dim=None, beta=1):
    def var_loss(self, y_hat, y, loss_fn, beta=1):
        """
            VAE loss function.

            Args:
                y_hat (torch tensor):   VAE predictions
                y     (torch tensor):   Targets
                batch_size (int):       Batch size
                inp_dim (int):          Input dimension
                beta (float):           beta-VAE parameter in the range [0, 1]

            Returns:
                The loss between the reconstructions (y_hat) and the target
                (y).
        """
        scale = y.shape[1] * y.shape[0]
        L = loss_fn(y_hat, y)
        KLD = -0.5 * sum(1 + self.std - self.mu.pow(2) - self.std.exp())
        KLD /= scale
        return L + beta * KLD

    def forward(self, x):
        """
            Forward method of LVAE Class.

            Args:
                x (Torch Tensor): Input of shape (BSize, SeqLen, #features)

            Returns:
                output of size (BatchSize, SeqLen, #features)
        """
        res = self.encoder(x)
        res = self.reparametrize(res)
        res = self.decoder(res)
        return res
