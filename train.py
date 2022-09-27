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
from torch import nn, cuda, device
from torch import optim, save
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
          model_path='./results/lstm_forecast_model.pt',
          data_path='./data/sinusoidal.npy',
          model='cnn'):
    """

    """
    store = True
    crit_flag = 0
    net_flag = 0        # 0 CNN, 1 LSTM

    dev = device("cuda:0" if cuda.is_available() else "cpu")
    print("Running on", dev)

    if model == 'linear':
        print("Linear VAE")
        shape = (seq_len * num_features, 256)
        funs_enc = (nn.ReLU(inplace=True),)
        funs_dec = (nn.ReLU(inplace=True), nn.Sigmoid())
        net = LVAE(shape=shape,
                   ldim=16,
                   device=dev,
                   funs_enc=funs_enc,
                   funs_dec=funs_dec).to(dev)
        net_flag = 0
        crit_flag = 2
    elif model == 'cnn':
        print("CNN AE")
        net = CNNAE(in_channels=num_features, sequence_length=seq_len).to(dev)
        net_flag = 0
        crit_flag = 0
    elif model == 'causal':
        print("CausalCNN AE")
        net = CausalCNNAE(in_channels=num_features,
                          sequence_length=seq_len).to(dev)
        net_flag = 0
        crit_flag = 0
    elif model == 'lstm_ae':
        print("LSTM AE")
        net = LSTMAE(1, 1, 128, 16, 2, seq_len=seq_len, enc_dropout=0.2,
                     dec_dropout=0.5, dev=dev).to(dev)
        net_flag = 1
        crit_flag = 0
    elif model == 'lstm_vae':
        print("LSTM VAE")
        net = LSTMVAE(1, 1, 256, 64, 2, seq_len=seq_len, enc_dropout=0.5,
                      dec_dropout=0.5, dev=dev).to(dev)
        net_flag = 1
        crit_flag = 0
    elif model == 'cnn_vae':
        print("CNN VAE")
        net = CNNVAE(in_channels=num_features, sequence_length=seq_len,
                     dev=dev).to(dev)
        net_flag = 0
    elif model == 'causal_vae':
        print('CausalCNN VAE')
        net = CausalCNNVAE(in_channels=num_features,
                           sequence_length=seq_len, dev=dev).to(dev)
        net_flag = 0
        crit_flag = 1
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
                            shuffle=False,
                            drop_last=True)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lrate, weight_decay=1e-5)

    # Scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 300, gamma=0.1,
    #                                       last_epoch=-1)
    criterion = nn.MSELoss()

    loss_track = []
    for e in range(epochs):
        for data in dataloader:
            x, _, _ = data
            x = x.to(dev)
            if net_flag == 0:
                x = x.permute(0, 2, 1)

            if crit_flag == 2:
                x = x.view(-1, seq_len * num_features)
            optimizer.zero_grad()
            y_hat = net(x)

            if crit_flag == 0:
                loss = criterion(y_hat, x)
            else:
                loss = net.var_loss(y_hat, x, criterion)
            loss.backward()
            optimizer.step()
        # scheduler.step()

        if e % 20 == 0:
            loss_track.append(loss.item() / batch_size)
            lrate = optimizer.param_groups[-1]['lr']
            print("[%d] Loss: %f  %f" % (e, loss.item() / batch_size,
                                         lrate))
    print("[%d] Loss: %f  %f" % (e, loss.item() / batch_size, lrate))
    if store is True:
        save(net, model_path)
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
                        metavar='N',
                        help='learning rate for training (default: 2e-3)')
    parser.add_argument('--seq-len', type=int, default=20, metavar='N',
                        help='input sequence length (default: 20)')
    parser.add_argument('--num-features', type=int, default=1, metavar='N',
                        help='input dimension (default: 1)')
    parser.add_argument('--data-path', type=str,
                        default='./data/sinusoidal.npy'
                                + 'data/livelo.npy',
                        metavar='N',
                        help='Data set full path')
    parser.add_argument('--model-path', type=str,
                        default='./results/cnn_vae_model.pt',
                        metavar='N',
                        help='Trained model full path')
    parser.add_argument('--model', type=str,
                        default='cnn',
                        metavar='N',
                        help='model type')
    args = parser.parse_args()
    train(args.epochs,
          args.batch_size,
          args.learning_rate,
          args.seq_len,
          args.num_features,
          args.model_path,
          args.data_path,
          args.model)
