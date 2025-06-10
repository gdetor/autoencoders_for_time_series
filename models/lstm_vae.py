# A Pytorch immplementation of an Variational Autoencoder with LSTM
# encoder/decoder.
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
from torch import nn
from torch import sum
from torch.autograd import Variable

from models.lstm_ae import encoder, decoder


class reparametrize(nn.Module):
    """
    Reparametrization trick class. This class implements the reparametrization
    trick as in Kingma and Welling "Auto-Encoding Variational Bayes".
    """
    def __init__(self, hidden_dim, latent_dim):
        """
        Constructor of reparametrize class.

        Args:
            hidden_dim (int):   Number of units in the last hidden layer
            latent_dim (int):   The dimension of latent space

        Returns:

        """
        super(reparametrize, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.h2mean = nn.Linear(hidden_dim, latent_dim)
        self.h2var = nn.Linear(hidden_dim, latent_dim)

        nn.init.xavier_normal_(self.h2mean.weight)
        nn.init.xavier_normal_(self.h2var.weight)

    def forward(self, x):
        """
        Forward method of reparametrize class. Estimates the latent variable
        Z via eps as in Kingma and Welling (see above).

        Args:
            x (torch tensor):   The latent space (before the reparametrization)

        Returns:
            The latent space Z after reparametrization.
        """
        if self.training:
            self.mu = self.h2mean(x)
            self.var = self.h2var(x)
            self.std = self.var.exp_()
            eps = Variable(self.std.data.new(
                self.std.size()).normal_()).to(x.device)
            return eps.mul(self.std).add_(self.mu)
        else:
            self.mu = self.h2mean(x)
            return self.mu


class LSTMVAE(nn.Module):
    """
        LSTM Variational Autoencoder Class
    """
    def __init__(self,
                 in_dim=1,
                 out_dim=1,
                 hidden_dim=16,
                 latent_dim=8,
                 n_layers=1,
                 seq_len=1,
                 enc_dropout=0.0,
                 dec_dropout=0.0):
        """
            Constructor of LSTM Encoder class.

            Args:
                in_dim (int):       Number of input features
                out_dim (int):      Number of output features
                hidden_dim (int):   Number of hidden units
                n_layers (int):     Number of LSTM layers
                seq_len (int):      Sequence length
                enc_dropout (float):  Encoder dropout probability [0, 1]
                enc_dropout (float):  Decoder dropout probability [0, 1]

            Returns:
        """
        super(LSTMVAE, self).__init__()
        self.seq_len = seq_len
        self.ddim = out_dim

        # Encoder
        self.encoder = encoder(in_dim, out_dim, hidden_dim, n_layers,
                               dropout=enc_dropout)

        # Reparametrization
        self.reparam = reparametrize(hidden_dim, latent_dim)

        # Decoder
        self.decoder = decoder(hidden_dim, hidden_dim, latent_dim, out_dim,
                               n_layers, seq_len, dropout=dec_dropout,
                               is_vae=True)

        # Annealing parameter (temperature)
        self.alpha = 0

    def forward(self, x):
        """
            Forward method of LSTM AE.

            Args:
                x (torch tensor): Input tensor (batch_size, seq_length,
                                                num_features)

            Returns:
                Output tensor (batch_size, sequence_length, num_features)
        """
        out, c = self.encoder(x)
        z = self.reparam(out)
        out = self.decoder(z, c)
        return out

    def var_loss(self, y_hat, y, loss_fn, beta=1):
        """
        Variational Loss (loss function + beta * KLD).

        Args:
            y_hat (torch tensor):   Tensor with predictions
            y (torch tensor):       Tensor with targets
            loss_fn (func):         Loss function (e.g, MSE, BCE, etc)
            beta (float):           beta parameter (b=0 => AE, b=1 => VAE)

        Returns:
            The variational loss (scalar) [batch_size, ndim], the loss_fn,
            and the KL divergence.
        """
        L = loss_fn(y_hat, y)
        scale = y.shape[2] * y.shape[1]
        KLD = -0.5 * sum(1 + self.reparam.std - self.reparam.mu.pow(2) -
                         self.reparam.std.exp())
        KLD /= scale
        return L + beta * KLD

    def var_loss_anneal(self, y_hat, y, loss_fn, epoch=0, beta=1):
        """
        Annealed variational Loss (loss function + beta * KLD). Anneals the
        variational loss based on number of epochs. Helps improving the
        training of a VAE.

        Args:
            y_hat (torch tensor):   Tensor with predictions
            y (torch tensor):       Tensor with targets
            loss_fn (func):         Loss function (e.g, MSE, BCE, etc)
            epoch (int):            Number of epoch
            beta (float):           beta parameter (b=0 => AE, b=1 => VAE)

        Returns:
            The variational loss (scalar) [batch_size, ndim], the loss_fn,
            and the KL divergence.
        """
        scale = y.shape[2] * y.shape[1]
        L = loss_fn(y_hat, y)
        KLD = -0.5 * sum(1 + self.reparam.std - self.reparam.mu.pow(2) -
                         self.reparam.std.exp())
        KLD /= scale
        if epoch > 40:
            tmp_alpha = min(self.alpha + 1./40, 1)
            self.alpha = tmp_alpha
        return L + self.alpha * beta * KLD, L, KLD
