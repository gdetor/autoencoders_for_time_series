# Autoencoders for time series

This repository provides a collection of autoencoders (AEs) and Variational AEs
for time series data sets implemented in Pytorch.
Autoencoders have several applications the time series field, such as
  - classification [1]
  - anomaly detection [2],
  - and time series prediction [3].


### Available Autoencoders

In this repository you will find the following models:
  - **Linear** AE with feed-forward encoder/decoder.
  - **CNN** AE with convolution neural network as encoder/decoder.
  - **Causal** AE with causal convolution encoder/decoder (dilated convolutions)
  - **LSTM** AE with LSTM encoder/decoder.

For each case the variational AE [4] is also available in the directory `models/`.



### How to run

You can run any of the seven different models by just passing the necessary 
arguments to the python script `train.py`.

```bash
$ python3 train.py --epochs XX --batch-size XX --learning-rate X.XXXX --seq-len X --num-features X --data-path PATH_TO_DATAFILE --model-path PATH_TO_MODEL.pt --model MODEL_TYPE
```

It's up to you to define the number of epochs, batch size, learning rate, 
the full path and file name of the data file as well as the full path and 
the chosen file name for the model (where to save the Pytorch model for later
use). Furthermore, you can specify if the input time series is univariate 
(number of features is 1) or multivariate (then the number of features is > 1).
You are also responsible for setting the number of past (historical) points
you'd like to use in the forecasting--prediction (meaning how many data points
the model will look into the past to predict one point in the future).

Moreover, you can test any trained model by calling the script `test.py` and
passing the corresponding to the model arguments.

```bash
$ python3 test.py --batch-size 32 --seq-len 12 --num-features 1 --mlp-flag 0 --lstm-flag 1 --data-path ./data/sinusoidal.npy --model-path ./tune_dir/causal.pt

```
The agruments that are similar with the training script have the exact same 
meaning. However, you have to tell to the script if your testing model is an
LSTM (set the flag --lstm-flag to 1) and if your model is an MLP (set the
flag --mlp-flag to 1).



### How to tune the hyperparameters

In this repository you will find a script named `tune_hyperparameters.py` that
provides all the necessary means to perform a search for your model's hyperparameters.
The script is heavily based on the Python Ray Framework. 

```bash
$ python3 tune_hyperparams.py --model-type MODEL_TYPE --data-path FULL_PATH_TO_DATA_FILE --store-data-path FULL_PATH_TO_WHERE_THE_RESULTS_WILL_BE_STORED

```
You will have to provide the model type, the path to the input data file and
the full path to the directory where results will be stored.


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
