# Titanic Binary Classifier

Titanicbc is a simple interface for training binary classification Networks 
or using a pre_trained network to output predictions for the Kaggle Titanic Dataset.

Later versions will expand the package to contain more flexible interfaces and networks for other classic datasets.

### Installation

You can install Titanicbc from PyPI

___
pip install Titanicbc

___

## How to use

___

Titanicbc provides a simple interface for training and using pre-trained networks via the config.yaml file.

The options are presented below in the following format;

option number. Key (value options) 

1. data_load (csv, database) - csv loads csv files stored in the package as train and test, database loads files from
train and test tables within a postgresql database.

2. num_columns (Integer) - number of columns in the train dataset.

3. train_new (True, False) - If true, a new neural network will be trained and overwrite trained_model.pth. If False the model parameters saved
in trained_model.pth will be loaded and used for predictions.

4. hidden_dim (Integer) - Number of neurons in each of the 3 hidden layers within the network.

5. num_epochs (Integer) - Number of passes the network will make over the training data when training a new model

6. learning_rate (float) - Parameter multiplied to the weight updates during stochastic gradient descent

___
