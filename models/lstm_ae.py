# A Pytorch implementation of an Autoencoder with LSTM Encoder/Decoder modules.
# Copyright (C) 2022  Georgios Is. Detorakis
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
from torch import nn, stack
from torch import randn, zeros


class Identity(nn.Module):
    """
        Identity class, implements an identity layer.
    """
    def __init__(self):
        """
        Constructor of Identity class.

        Args:
            None

        Returns:
        """
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward method of Identity class.

        Args:
            x (torch tensor):   Input tensor

        Returns:
            A torch tensor identical to input.
        """
        return x


class encoder(nn.Module):
    """
        LSTM Encoder class
    """
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, dropout=0):
        """
            Constructor of LSTM Encoder class.

            Args:
                in_dim (int):       Number of input features
                out_dim (int):      Number of output features
                hidden_dim (int):   Number of hidden units
                n_layers (int):     Number of LSTM layers
                dropout (float):    Dropout probability [0, 1]

            Returns:
        """
        super(encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout)

        # Initialize weights and biases
        for layer in self.lstm._all_weights:
            for la in layer:
                if 'weight' in la:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(la).data)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = self.lstm.__getattr__(name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def forward(self, x):
        """
            Forward method of LSTM Encoder.

            Args:
                x (torch tensor): Input tensor (batch_size, seq_length,
                                                num_features)

            Returns:
                Output tensor (batch_size, sequence_length, num_features)
        """
        batch_size = x.shape[0]
        self.h0 = zeros(self.n_layers*1,
                        batch_size,
                        self.hidden_dim,
                        requires_grad=True).to(x.device)
        self.c0 = zeros(self.n_layers*1,
                        batch_size,
                        self.hidden_dim,
                        requires_grad=True).to(x.device)
        out, (h_end, c_end) = self.lstm(x, (self.h0, self.c0))
        return h_end[-1, :, :], self.c0


class decoder(nn.Module):
    """
        LSTM Decoder class
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, n_layers,
                 seq_len, dropout=0, is_vae=False):
        """
            Constructor of LSTM Encoder class.

            Args:
                in_dim (int):       Number of input features
                out_dim (int):      Number of output features
                hidden_dim (int):   Number of hidden units
                n_layers (int):     Number of LSTM layers
                seq_len (int):      Sequence length
                dropout (float):    Dropout probability [0, 1]
                dev (torch device): Device to upload the model

            Returns:
        """
        super(decoder,  self).__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_vae = is_vae

        # LSTM model
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        # Hidden to output linear layer
        self.h2o = nn.Linear(hidden_dim, out_dim)

        # Latent space to hidden
        if self.is_vae:
            self.latent2h = nn.Linear(latent_dim, hidden_dim)
        else:
            self.latent2h = Identity()

        # Weights and biases initialization
        for layer in self.lstm._all_weights:
            for la in layer:
                if 'weight' in la:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(la).data)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = self.lstm.__getattr__(name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, x, c):
        """
            Forward method of LSTM Decoder.

            Args:
                x (torch tensor): Input tensor (batch_size, seq_length,
                                                num_features)

            Returns:
                Output tensor (batch_size, sequence_length, num_features)
        """
        self.decoder_inp = randn(x.shape[0], self.seq_len, self.hidden_dim,
                                 requires_grad=True).to(x.device)

        h = self.latent2h(x)
        h_0 = stack([h for _ in range(self.n_layers)])
        out, _ = self.lstm(self.decoder_inp, (h_0, c))
        out = self.h2o(out)
        return out


class LSTMAE(nn.Module):
    """
        LSTM Autoencoder Class
    """
    def __init__(self,
                 in_dim=1,
                 out_dim=1,
                 hidden_dim=1,
                 latent_dim=1,
                 n_layers=1,
                 seq_len=1,
                 enc_dropout=0,
                 dec_dropout=0):
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
        super(LSTMAE, self).__init__()
        self.seq_len = seq_len
        self.ddim = out_dim

        # Encoder
        self.encoder = encoder(in_dim,
                               out_dim,
                               hidden_dim,
                               n_layers,
                               dropout=enc_dropout)
        # Decoder
        self.decoder = decoder(hidden_dim,
                               hidden_dim,
                               latent_dim,
                               out_dim,
                               n_layers,
                               seq_len,
                               dropout=dec_dropout)

        # Latent space
        self.h2l = nn.Linear(hidden_dim, latent_dim)
        self.l2h = nn.Linear(latent_dim, hidden_dim)

        # Output nonlinearity
        self.relu = nn.ReLU()

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
        # out = out.view(-1, out.shape[1])
        z = self.h2l(out)
        out = self.l2h(z)
        # out = out.view(-1, out.shape[1])
        out = self.decoder(out, c)
        return out
