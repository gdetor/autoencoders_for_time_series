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


### How to tune the hyperparameters


### Dependencies

To run the training/testing scripts you will need the following dependencies:
  * Numpy
  * Matplotlib
  * Sklearn
  * Torch (Pytorch, Torchvision)
  * Ray Tune


### References
  1. First
  2. Second
  3. Third
  4. Fourth
