# Autoencoders for time series

This repository provides a collection of autoencoders (AEs) 
and Variational AEs for time series data sets implemented in Pytorch.
Autoencoders have several applications in the time series field, such
as
  - classification [1]
  - anomaly detection [2],
  - and time series prediction [3].


### Available Autoencoders

In this repository, you will find the following models in the `models/`
directory:
  - **Linear** AE with feed-forward encoder/decoder.
  - **CNN** AE with convolution neural network as encoder/decoder.
  - **Causal** AE with causal convolution encoder/decoder (dilated convolutions)
  - **LSTM** AE with LSTM encoder/decoder.

For each case above, a variational AE [4] is also available in the directory
`models/`. In the directory `data_loader/`, you can find the  TimeseriesLoader
class (see [here](https://github.com/gdetor/pytorch_timeseries_loader) for more details),
and in the directory `data/`, you can find two data files (a sinusoidal
signal and the female's births dataset).


### How to run

You can run any of the seven different models by just passing the necessary 
arguments to the python script `train.py`. Before you run the scripts, though, you should
create a `./results/` directory.

```bash
$ mkdir results
$ python3 train.py --epochs XX --batch-size XX --learning-rate X.XXXX --seq-len X --num-features X --data-path PATH_TO_DATAFILE --model MODEL_TYPE
```

It's up to you to define the number of epochs, batch size, learning rate, 
the full path and file name of the data file, as well as the full path and 
the chosen file name for the model (where to save the Pytorch model for later
use). Furthermore, you can specify if the input time series is univariate 
(number of features is 1) or multivariate (then the number of features is > 1).
You are also responsible for setting the number of past (historical) points
you'd like to use in the forecasting--prediction (meaning how many data points
the model will look into the past to predict one point in the future).

The `MODEL_TYPE` should be one of the following options:
  - "cnn"
  - "cnn_vae"
  - "mlp_vae" (linear)
  - "lstm_ae"
  - "lstm_vae"
  - "causal_ae"
  - "causal_vae"

Moreover, you can test any trained model by calling the script `test.py` and
passing the corresponding model arguments.

```bash
$ python3 test.py --batch-size 32 --seq-len 12 --num-features 1 --data-path ./data/sinusoidal.npy --model MODEL_TYPE

```
The arguments similar to the training script have the same meaning.


### Dependencies

The following dependencies are necessary for executing the train/test and tune scripts:
  * Numpy
  * Matplotlib
  * Sklearn
  * Torch (Pytorch, Torchvision)
  * Ray (1.8.0)


### References
  1. Mehdiyev, Nijat, Johannes Lahann, Andreas Emrich, David Enke, Peter Fettke, and Peter Loos.
     ``Time series classification using deep learning for process planning: A case from the process industry.``
     Procedia Computer Science 114 (2017): 242-249.
  2. Tziolas, Theodoros, Konstantinos Papageorgiou, Theodosios Theodosiou, Elpiniki Papageorgiou, Theofilos Mastos, and Angelos Papadopoulos.
    ``Autoencoders for anomaly detection in an industrial multivariate time series dataset.``
    Engineering Proceedings 18, no. 1 (2022): 23.
  3. https://www.kaggle.com/code/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders/notebook
  4. Kingma, Diederik P., and Max Welling. ``An introduction to variational autoencoders.``
     Foundations and Trends® in Machine Learning 12, no. 4 (2019): 307-392.
