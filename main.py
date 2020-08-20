from Model_Code import Binary_Network

Binary_Network.train_new_model('Data/train.csv')

# load models
# with open('config.yaml') as file:
#     model_parameters = yaml.load(file, Loader=yaml.FullLoader)
model = Binary_Network.Binary_Network(7, 7)
model = Binary_Network.load_models('trained_model.pth', model)
Binary_Network.predict(model, 'Data/test.csv')
# Make sure basic config of network is the same - config.yaml and then call load models then predict

# Create load model function
# Learn how to put parameters into yaml/json
#