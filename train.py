# Implements of the training function for all the Autoencoders of this
# repository.
# Copyright (C)  2020 Georgios Is. Detorakis
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
import argparse
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader.timeseries_loader import TimeseriesLoader

from models.linear_vae import LVAE
from models.cnn_ae import CNNAE
from models.causalcnn_ae import CausalCNNAE
from models.lstm_ae import LSTMAE
from models.lstm_vae import LSTMVAE
from models.cnn_vae import CNNVAE
from models.causalcnn_vae import CausalCNNVAE
import matplotlib.pylab as plt


def train(epochs,
          batch_size,
          lrate,
          seq_len,
          num_features,
          is_vae_on=False,
          model_path='./results/lstm_forecast_model.pt',
          data_path='./data/sinusoidal.npy',
          model='cnn'):
    """
        Trains one of the following autoencoders: Linear VAE, CNN AE, CNN VAE,
        Causal CNN AE, Causal CNN VAE, LSTM AE, LSTM VAE.
        The type of model can be one of the following: 'linear', 'cnn',
        'cnn_vae', 'causal', 'causal_vae', 'lstm_ae', 'lstm_vae'.

        Args:
            epochs (int)        Number of training epochs
            batch_size (int)    Batch size
            lrate (float)       Learning rate
            seq_len (int)       Sequence length for histortical (past) data
            num_features (int)  Dimension of features (1 for univariate time
            series)
            is_vae_on (str)     Engages VAE's loss function
            model_path (str)    Where to store the model (.pt file)
            data_path (str)     Where the training data are stored
            model (str)         Model type (see above)

        Returns: void
    """
    store = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    if model == 'mlp_vae':
        print("MLP VAE")
        shape = (seq_len * num_features, 256)
        # Define the nonlinear activation functions for the MLP
        funs_enc = (nn.ReLU(inplace=True),)
        funs_dec = (nn.ReLU(inplace=True), nn.Sigmoid())

        net = LVAE(shape=shape,
                   ldim=16,
                   funs_enc=funs_enc,
                   funs_dec=funs_dec).to(device)
    elif model == 'cnn':
        print("CNN AE")
        net = CNNAE(in_channels=num_features,
                    sequence_length=seq_len).to(device)
    elif model == 'cnn_vae':
        print("CNN VAE")
        net = CNNVAE(in_channels=num_features,
                     sequence_length=seq_len).to(device)
    elif model == 'causal':
        print("CausalCNN AE")
        net = CausalCNNAE(in_channels=num_features,
                          sequence_length=seq_len).to(device)
    elif model == 'causal_vae':
        print('CausalCNN VAE')
        net = CausalCNNVAE(in_channels=num_features,
                           sequence_length=seq_len).to(device)
    elif model == 'lstm_ae':
        print("LSTM AE")
        net = LSTMAE(in_dim=1,
                     out_dim=1,
                     hidden_dim=128,
                     latent_dim=16,
                     n_layers=2,
                     seq_len=seq_len,
                     enc_dropout=0.1,
                     dec_dropout=0.1).to(device)
    elif model == 'lstm_vae':
        print("LSTM VAE")
        net = LSTMVAE(in_dim=1,
                      out_dim=1,
                      hidden_dim=256,
                      latent_dim=64,
                      n_layers=2,
                      seq_len=seq_len,
                      enc_dropout=0.1,
                      dec_dropout=0.1).to(device)
    else:
        print("No model specified!")
        sys.exit(-1)

    # Dataset and dataloader
    ts = TimeseriesLoader(data_path=data_path,
                          sequence_len=seq_len,
                          scale=True,
                          standarize=False,
                          train=True,
                          entire_seq=False,
                          data_split_perc=0.7)
    dataloader = DataLoader(ts,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lrate,
                                  weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                300,
                                                gamma=0.1,
                                                last_epoch=-1)
    criterion = nn.MSELoss()

    loss_track = []
    for e in range(epochs):
        for data in dataloader:
            x, _, _ = data
            x = x.to(device)

            optimizer.zero_grad()

            x_hat = net(x)

            if is_vae_on is False:
                loss = criterion(x_hat, x)
            else:
                loss = net.var_loss(x_hat, x, criterion)

            loss.backward()
            optimizer.step()
        scheduler.step()

        if e % 20 == 0:
            loss_track.append(loss.item() / batch_size)
            lrate = optimizer.param_groups[-1]['lr']
            print("[%d] Loss: %f  %f" % (e, loss.item() / batch_size,
                                         lrate))
    print("[%d] Loss: %f  %f" % (e, loss.item() / batch_size, lrate))
    if store is True:
        torch.save(net, model_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(loss_track, 'f'), 'k', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    seq_len = 40
    num_features = 1
    epochs = 200
    lrate = 1e-4
    batch_size = 32
    parser = argparse.ArgumentParser(description='LSTM Forecasting Test')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs for training (default: 300)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=2e-3,
                        help='learning rate for training (default: 2e-3)')
    parser.add_argument('--seq-len', type=int, default=20, metavar='N',
                        help='input sequence length (default: 20)')
    parser.add_argument('--num-features', type=int, default=1, metavar='N',
                        help='input dimension (default: 1)')
    parser.add_argument('--data-path', type=str,
                        default='./data/sinusoidal.npy',
                        help='Data set full path')
    parser.add_argument('--model-path', type=str,
                        default='./results/cnn_vae_model.pt',
                        help='Trained model full path')
    parser.add_argument('--model', type=str,
                        default='cnn',
                        help='model type (mlp_vae, causal, causal_vae,\
                                lstm_ae, lstm_vae, cnn_vae, causal_vae)')

    args = parser.parse_args()

    is_vae_on = False
    if "vae" in args.model:
        is_vae_on = True

    train(args.epochs,
          args.batch_size,
          args.learning_rate,
          args.seq_len,
          args.num_features,
          is_vae_on=is_vae_on,
          model_path="./results/"+args.model,
          data_path=args.data_path,
          model=args.model)
