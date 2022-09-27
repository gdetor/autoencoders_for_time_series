# In this script you will find the test function for validation the training
# of the Autoencoders found in this repository.
# Copyright (C)  2020  Georgios Is. Detorakis
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
import numpy as np
import matplotlib.pylab as plt

from torch import device
from torch import no_grad, load
from torch.utils.data import DataLoader

from data_loader.timeseries_loader import TimeseriesLoader


def test(seq_len=20,
         batch_size=1,
         num_features=1,
         is_mlp_on=0,
         is_lstm_on=False,
         model_path='./results/cnn_vae_model.pt',
         data_path='./data/sinusoidal.npy'):
    """
        Tests a trained autoencoder. The autoencoder's model should be stored
        in a local directory (e.g., ./tmp/). The function will plot some of
        the results.

         Args:
             seq_len (int)       Sequence length for histortical (past) data
             batch_size (int)    Batch size
             num_features (int)  Dimension of features (1 for univariate time
             series)
             is_mlp_on (int)     1 means the current model is a Linear (MLP
                                 AE/VAE)
                                 0 for any other model
             is_lstm_on_on (int)    1 means the current model is an LSTM AE/VAE
                                 0 for any other model
             model_path (str)    Where to store the model (.pt file)
             data_path (str)     Where the training data are stored

         Returns: void
    """
    print("Testing model:", model_path)
    print("on data: ", data_path)
    dev = device("cuda:0")
    net = load(model_path)
    if is_lstm_on == 1:
        net.encoder.flag = 0
        net.decoder.flag = 0
    print(net)

    ts = TimeseriesLoader(data_path=data_path,
                          sequence_len=seq_len,
                          scale=True,
                          standarize=False,
                          train=False,
                          entire_seq=False,
                          data_split_perc=0.7)
    dataloader = DataLoader(ts,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True)

    net.eval()
    with no_grad():
        X, Y = [], []
        for i, data in enumerate(dataloader):
            x, _, _ = data
            x = x.to(dev)
            if is_mlp_on == 0 and is_lstm_on == 0:
                x = x.permute(0, 2, 1)
            if is_mlp_on == 1:
                x = x.view(-1, seq_len * num_features)
            res = net(x)
            X.append(x.detach().cpu().numpy())
            Y.append(res.detach().cpu().numpy())
    X = np.array(X)
    Y = np.array(Y)
    # np.save("./tune_dir/test_original", X)
    # np.save("./tune_dir/test_reconstr", Y)

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    plt.plot(X[0, :], 'm-o', lw=2, ms=10, label='Original')
    plt.plot(Y[0, :], 'k-x', lw=2, ms=10, label='Reconstruction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Forecasting Test')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--seq-len', type=int, default=20, metavar='N',
                        help='input sequence length (default: 20)')
    parser.add_argument('--num-features', type=int, default=1, metavar='N',
                        help='number of features (default: 1)')
    parser.add_argument('--mlp-flag', type=int, default=0, metavar='N',
                        help='MLP initialization flag (default: 0)')
    parser.add_argument('--lstm-flag', type=int, default=0, metavar='N',
                        help='LSTM initializaation flag (default: 0)')
    parser.add_argument('--data-path', type=str,
                        default='/home/gdetor/ergasia/tests/wgan_ts/'
                                + 'data/livelo.npy',
                        metavar='N',
                        help='Data set full path')
    parser.add_argument('--model-path', type=str,
                        default='./results/cnn_vae_model.pt',
                        metavar='N',
                        help='Trained model full path')

    args = parser.parse_args()
    test(seq_len=args.seq_len,
         batch_size=args.batch_size,
         num_features=args.num_features,
         is_mlp_on=args.dim_flag,
         is_lstm_on=args.lstm_flag,
         model_path=args.model_path,
         data_path=args.data_path)
