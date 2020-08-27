from Titanicbc import Binary_Network
import torch
import yaml
import pandas as pd
from importlib import resources as res

def main():
    ##### Set Pytorch Device #####

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #### Read in model parameters #####

    with res.open_binary('Titanicbc', 'config.yaml') as fp:
        model_parameters = yaml.load(fp, Loader=yaml.Loader)

    num_columns = 12
    train_new = model_parameters['train_new']
    hidden_dim = model_parameters['Binary_Network']['initialisations']['hidden_dim']
    learning_rate = model_parameters['Binary_Network']['optimiser']['learning_rate']
    epochs = model_parameters['Binary_Network']['num_epochs']

    input_dim = num_columns - 5

    ####Read in data from csv ####
    with res.open_binary('Titanicbc', 'train.csv') as train:
        train = pd.read_csv(train)
    with res.open_binary('Titanicbc', 'test.csv') as test:
        test = pd.read_csv(test)


    #### Main Script ####

    with res.path('Titanicbc', 'trained_model.pth') as m:
        model_path = m
    if train_new:
        model = Binary_Network.train_new_model(train, input_dim, hidden_dim, model_path, learning_rate, epochs)

    else:
        model = Binary_Network.Binary_Network(input_dim, hidden_dim)
        model = Binary_Network.load_models(model_path, model).to(device)

    Binary_Network.predict(model, test)

if __name__ == "__main__":
    main()