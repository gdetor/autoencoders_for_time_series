# This auxiliary scripts helps tune the hyperparameters of the autoencoders
# found in this repo.
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
from models.cnn_vae import CNNVAE
from models.causalcnn_vae import CausalCNNVAE
import matplotlib.pylab as plt

from ray import tune

rng = np.random
np.set_printoptions(threshold=np.inf)


def train(config):
    if len(sys.argv) != 3:
        print("Please provide the type of model and the data path!")
        exit(-1)

    # Fixed parameters
    store = True
    num_features = 1

    # Hyperparameters to tune
    epochs = config['epochs']
    lrate = config['lrate']
    batch_size = config['batch_size']
    seq_len = config['sequence_len']

    # Other parameters
    crit_flag = 0
    net_flag = 0        # 0 CNN, 1 LSTM

    # Choose the device on which we are going to run the tuning
    dev = device("cuda:0" if cuda.is_available() else "cpu")
    print("Running on", dev)

    # Choose the model
    if sys.argv[1] == 'linear':
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
    elif sys.argv[1] == 'cnn':
        print("CNN AE")
        net = CNNAE(in_channels=num_features, sequence_length=seq_len).to(dev)
        net_flag = 0
        crit_flag = 0
    elif sys.argv[1] == 'causal':
        print("CausalCNN AE")
        net = CausalCNNAE(in_channels=num_features,
                          sequence_length=seq_len).to(dev)
        net_flag = 0
        crit_flag = 0
    elif sys.argv[1] == 'lstm':
        net = LSTMAE(1, 1, 128, 16, 2, seq_len=seq_len, dev=dev).to(dev)
        net_flag = 1
        crit_flag = 0
    elif sys.argv[1] == 'vaecnn':
        net = CNNVAE(in_channels=num_features, sequence_length=seq_len,
                     dev=dev).to(dev)
        net_flag = 0
    elif sys.argv[1] == 'vaecausal':
        net = CausalCNNVAE(in_channels=num_features,
                           sequence_length=seq_len, dev=dev).to(dev)
        net_flag = 0
        crit_flag = 1
    else:
        print("No model specified!")
        sys.exit(-1)

    # Dataset and dataloader
    ts = TimeseriesLoader(data_path=sys.argv[2],
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

    if store is True:
        base = "./results/"
        save(net.state_dict(), base+"cnn_ae_st_indicators.pt")
        save(net, base+"cnn_ae_model_indicators.pt")
    print("[%d] Loss: %f  %f" % (e, loss.item() / batch_size, lrate))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(loss_track, 'f'), 'k', alpha=0.6)
    returned_loss = loss.item() / batch_size

    tune.report(mean_loss=returned_loss)


if __name__ == '__main__':
    search_space = {"epochs": tune.choice([50, 100, 150, 200]),
                    "lrate": tune.uniform(1e-6, 1e-3),
                    "sequence_len": tune.choice([4, 8, 12, 24]),
                    "batch_size": tune.choice([8, 16, 32, 64])}

    optimization = tune.run(config=search_space,
                            num_samples=10,
                            metric="mean_loss",
                            mode="min",
                            raise_on_failed_trial=True,
                            local_dir="./tune_dir",
                            resources_per_trial={"cpu": 3, "gpu": 1})
