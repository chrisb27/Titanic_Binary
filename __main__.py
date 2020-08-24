from Titanicbc import Binary_Network
import torch
import yaml
import pandas as pd

def main():
    ##### Set Pytorch Device #####

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #### Read in model parameters #####

    with open('config.yaml') as file:
        model_parameters = yaml.load(file, Loader=yaml.FullLoader)

    train_new = model_parameters['train_new']
    num_columns = model_parameters['num_columns']
    input_dim = model_parameters['Binary_Network']['initialisations']['input_dim']
    hidden_dim = model_parameters['Binary_Network']['initialisations']['hidden_dim']
    model_path = model_parameters['Binary_Network']['initialisations']['model_save_path']
    data_load = model_parameters['data_load']

    ####Read in data from database or csv ####
    if data_load == 'csv':
        train = pd.read_csv('Data/train.csv')
        test = pd.read_csv('Data/test.csv')

    elif data_load == 'database':
        train, test = Binary_Network.retrieve_from_database(num_columns)


    #### Main Script ####

    if train_new:
        model = Binary_Network.train_new_model(train, num_columns-5, hidden_dim, model_path)

    model = Binary_Network.Binary_Network(input_dim, hidden_dim)
    model = Binary_Network.load_models(model_path, model).to(device)

    Binary_Network.predict(model, test)

if __name__ == "__main__":
    main()