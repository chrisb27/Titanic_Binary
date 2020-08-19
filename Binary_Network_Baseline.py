import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print(device)


class Binary_Network(nn.Module):
        def __init__(self, features, hidden_size):
            super(Binary_Network, self).__init__()
            self.linear1 = nn.Linear(features, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1)

            torch.nn.init.xavier_uniform_(self.linear1.weight)
            torch.nn.init.xavier_uniform_(self.linear2.weight)
            torch.nn.init.xavier_uniform_(self.output_layer.weight)

        def forward(self, inputs):
            relu = torch.nn.ReLU().to(device)
            sm = nn.Sigmoid().to(device)  # 1 Dimensional data

            inputs = inputs.cuda()

            op = self.linear1(inputs)
            op = relu(op)

            op = self.linear2(op)
            op = relu(op)

            op = self.output_layer(op)
            op = sm(op)

            op = op.to(device)

            return op[0]


class TitanDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.FloatTensor(features.values).to(device)
        self.y = torch.FloatTensor(labels.values).to(device)

    def __len__(self):
        return (len(self.x))

    def __getitem__(self, idx):
        data = self.x[idx], self.y[idx]

        return data

def prep_train():
    train = pd.read_csv('Titanic_Binary/train.csv')
    test = pd.read_csv('Titanic_Binary/test.csv')
    train.set_index('PassengerId', inplace=True)
    train = train.drop(columns=['Name', 'Cabin', 'Ticket'])

    mapping = {'S': 2, 'Q': 1, 'C': 0}
    train['Embarked'] = train['Embarked'].str.upper().map(mapping)
    sex_mapping = {'male': 1, 'female': 0}
    train['Sex'] = train['Sex'].str.lower().map(sex_mapping)

    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
    train = train.dropna()
    train['Embarked'] = train['Embarked'].astype(int)

    print('dataset prepped')
    print(train.info())

    return train, test

def split_datasets(dataset,val_split):
    total_points = dataset.shape[0]
    fraction = int(total_points - (val_split*total_points))
    train = dataset[:fraction]
    validation = dataset[fraction:]

    print('train size:', train.shape)
    print('validation size:', validation.shape)
    train_features = train.drop(columns=['Survived'])
    train_labels = train['Survived']

    val_features = validation.drop(columns=['Survived'])
    val_labels = validation['Survived']

    return train_features, train_labels, val_features, val_labels

def create_datasets(train_features, train_labels, val_features, val_labels):
    train_data = TitanDataset(train_features, train_labels)
    val_data = TitanDataset(val_features, val_labels)

    return train_data, val_data

def prep_loaders(train_data, train_batch_size, val_data, val_batch_size):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

    return trainloader, valloader

def run_model(model, dataloader, num_epochs):
    model.train()
    running_loss = []
    for epoch in range(num_epochs):  # no. of epochs
        epoch_loss = 0
        correct = 0
        total_points = 0
        for data in dataloader:
            # print(data[0])
            # print(data[1])
            # data and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimiser.zero_grad()
            outputs = model(inputs)
            # print('outputs:', outputs, 'labels:', labels)
            basic_loss = criterion(outputs.float(), labels.float())
            # propagate the loss backward
            basic_loss.backward()
            # update the gradients
            optimiser.step()
            # print(basic_loss.item())
            epoch_loss += basic_loss.item()
            # print(basic_epoch_loss)
            output = (outputs > 0.5).float()
            if output == labels:
                correct += 1
            total_points += 1

        running_loss.append(epoch_loss)

        print(correct)
        print(total_points)
        print("Number_Epoch {}/{}, Total_Epoch_Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, num_epochs,
                                                                                      epoch_loss,
                                                                                      correct / total_points))

    return running_loss

def check_parameters(model):
    for parameter in model.parameters():
        print(parameter)

def batch_accuracy(inputs, labels, model):
    x, y = inputs, labels
    y_pred = (model(x)).type(torch.FloatTensor).to(device)
    y = y.unsqueeze(1)
    y_pred = (y_pred>0.5).float()
    correct = (y_pred == y).type(torch.FloatTensor).to(device)
    return correct.sum() # Returns the number of correct instances per batch

def accuracy(inputs, labels, model):

    output = model(inputs)
    output = (output > 0.5).float()
    if output == labels:
        return 1
    else:
        return 0

def evaluate_model(model, dataloader, val_labels, batch=False):
    model.eval()
    single_instance_acc = 0
    val_batch_acc = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            if batch:
                val_batch_acc.append(batch_accuracy(inputs, labels, model))
            else:
                single_instance_acc += accuracy(inputs, labels, model)

    if batch:
        total_correct = sum(val_batch_acc)
        print('Correct', total_correct.item(), 'Total', len(val_labels))
        print('Validation Accuracy', total_correct / len(val_labels))
    else:
        total_correct = single_instance_acc
        print('Correct', total_correct, 'Total', len(val_labels))
        print('Validation Accuracy', total_correct / len(val_labels))

dataset, _ = prep_train()
train_features, train_labels, val_features, val_labels = split_datasets(dataset, 0.1)
train_data, val_data = create_datasets(train_features, train_labels, val_features, val_labels)
trainloader, valloader = prep_loaders(train_data, 1, val_data, 1)

model = Binary_Network(7, 49).to(device)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)
running_loss = run_model(model, trainloader, 270)
evaluate_model(model, valloader, val_labels, False)
plt.plot(running_loss)
plt.show()