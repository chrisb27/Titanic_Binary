# Titanicbc

Titanicbc is a simple interface for training neural networks with custom hyper-parameters. The current version allows
training a binary classifier network for the famous Kaggle Titanic dataset. 

The aim of this package is to allow those with little or no neural network coding experience to learn how different hyper-parameter 
combinations affect neural network training. The package also includes a pre-trained neural network for demonstrating how networks make predictions
once trained.

Later versions will expand the package to contain more flexible interfaces and networks for other classic datasets,
including image and text datasets with convolutional and recurrent neural networks.

### Installation

You can install Titanicbc from PyPI

___
***pip install Titanicbc***

___

## How to use

___

Titanicbc provides a simple interface for training and using pre-trained networks via the config.yaml file.

The config.yaml file is included in the Python site-packages folder for Titanicbc. To find the python site-packages on
your machine run ***python -m site***. Once in site-packages, select the Titanicbc folder.

Once hyper-parameters have been set using config.yaml, simply run ***python -m Titanicbc*** from the command line or terminal to train a network or make 
predictions (depending on the value of train_new in config.yaml). The predictions made by the new or existing model will be saved into the same location in
site-packages/Titanicbc as output.csv.

___

The options for config.yaml are presented below in the following format;

***option number. Key (value options)*** 

1. train_new (True, False) - If true, a new neural network will be trained and overwrite trained_model.pth. If False the model parameters saved
in trained_model.pth will be loaded and used for predictions.

2. hidden_dim (Integer) - Number of neurons in each of the 3 hidden layers within the network.

3. num_epochs (Integer) - Number of passes the network will make over the training data when training a new model.

4. learning_rate (float) - Parameter multiplied to the weight updates during stochastic gradient descent. Currently only the Adam optimiser is used.

5. weight_init (uniform, xavier) - Tells the network which type of initialisation to use for the model weights. Xavier is currently recommended

___
