import sys
from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QPainter, QFont, QPen, QBrush
from PySide2.QtWidgets import (QAction, QApplication, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QFormLayout)
from Titanicbc import Binary_Network
import torch
import yaml
import pandas as pd
from importlib import resources as res


class Widget(QWidget):
    def __init__(self, device, train, test, input_dim):
        QWidget.__init__(self)

        self.device = device
        self.train = train
        self.test = test
        self.input_dim = input_dim

        with res.open_binary('Titanicbc', 'config.yaml') as fp:
            model_parameters = yaml.load(fp, Loader=yaml.Loader)
        print(model_parameters)

        train_new_current = model_parameters['train_new']
        hidden_dim_current = model_parameters['Binary_Network']['initialisations']['hidden_dim']
        learning_rate_current = model_parameters['Binary_Network']['optimiser']['learning_rate']
        num_epochs_current = model_parameters['Binary_Network']['num_epochs']
        weight_init_current = model_parameters['Binary_Network']['initialisations']['weight_init']
        weight_decay_current = model_parameters['Binary_Network']['optimiser']['weight_decay']

        # layout

        ## Read in current values from config.yaml as the default values in QForm
        self.layout = QFormLayout()

        self.train_new = QLineEdit(str(train_new_current))
        self.train_new_label = QLabel("Train New")

        self.num_epochs = QLineEdit(str(num_epochs_current))
        self.num_epochs_label = QLabel("Number of Epochs")

        self.learning_rate = QLineEdit(str(learning_rate_current))
        self.learning_rate_label = QLabel("Learning Rate")

        self.weight_decay = QLineEdit(str(weight_decay_current))
        self.weight_decay_label = QLabel("Weight Decay")

        self.weight_init = QLineEdit(str(weight_init_current))
        self.weight_init_label = QLabel("Weight Initialisation")

        self.hidden_dim = QLineEdit(str(hidden_dim_current))
        self.hidden_dim_label = QLabel("Hidden Layers Dimension")

        self.confirm = QPushButton("Confirm Network Configuration and train")
        self.predict = QPushButton("Predict using last trained model")
        self.quit = QPushButton("Quit")

        self.layout.addRow(self.train_new_label, self.train_new)
        self.layout.addRow(self.num_epochs_label, self.num_epochs)
        self.layout.addRow(self.learning_rate_label, self.learning_rate)
        self.layout.addRow(self.weight_decay_label, self.weight_decay)
        self.layout.addRow(self.hidden_dim_label, self.hidden_dim)
        self.layout.addRow(self.weight_init_label, self.weight_init)
        self.layout.addWidget(self.confirm)
        self.layout.addWidget(self.predict)
        self.layout.addWidget(self.quit)

        # Set the layout to the QWidget
        self.setLayout(self.layout)

        # Signals and Slots
        self.quit.clicked.connect(self.quit_application)
        self.confirm.clicked.connect(self.confirm_configuration)

        ## Execution here

    @Slot()
    def confirm_configuration(self):

        self.train_new = self.train_new.text()
        self.num_epochs = self.num_epochs.text()
        self.learning_rate = self.learning_rate.text()
        self.weight_decay = self.weight_decay.text()
        self.weight_init = self.weight_init.text()
        self.hidden_dim = self.hidden_dim.text()

        ## read about passing values out of here into
        print("Network Configuration")
        print("Train New: {}".format(self.train_new))
        print("Number of Epochs: {}".format(self.num_epochs))
        print("Learning Rate: {}".format(self.learning_rate))
        print("Weight Decay: {}".format(self.weight_decay))
        print("Weight Initialisation: {}".format(self.weight_init))
        print("Hidden Layers Dimension: {}".format(self.hidden_dim))

        with res.open_binary('Titanicbc', 'config.yaml') as fp:
            model_parameters = yaml.load(fp, Loader=yaml.Loader)

        model_parameters['train_new'] = str(self.train_new)
        model_parameters['Binary_Network']['initialisations']['hidden_dim'] = int(self.hidden_dim)
        model_parameters['Binary_Network']['optimiser']['learning_rate'] = float(self.learning_rate)
        model_parameters['Binary_Network']['num_epochs'] = int(self.num_epochs)
        model_parameters['Binary_Network']['initialisations']['weight_init'] = str(self.weight_init) ## Read in Binary_Network
        model_parameters['Binary_Network']['optimiser']['weight_decay'] = float(self.weight_decay)

        with res.path('Titanicbc', 'config.yaml') as cf:
            path = cf

        with open(path, 'w') as outfile:
            yaml.dump(model_parameters, outfile, default_flow_style=False)
        print(model_parameters)

        with res.path('Titanicbc', 'trained_model.pth') as m:
            model_path = m
        model = Binary_Network.train_new_model(self.train, self.input_dim, self.hidden_dim, model_path, self.learning_rate,
                                               self.num_epochs, self.weight_decay).to(self.device)
        Binary_Network.predict(model, self.test)


    @Slot()
    def predict(self):
        with res.path('Titanicbc', 'trained_model.pth') as m:
            model_path = m

        model = Binary_Network.Binary_Network(self.input_dim, self.hidden_dim)
        model = Binary_Network.load_models(model_path, model).to(self.device)
        Binary_Network.predict(model, self.test)


    @Slot()
    def quit_application(self):
        QApplication.quit()


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("Neural Network Configuration")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self):
        QApplication.quit()

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    with res.open_binary('Titanicbc', 'train.csv') as train:
        train = pd.read_csv(train)
    with res.open_binary('Titanicbc', 'test.csv') as test:
        test = pd.read_csv(test)

    input_dim = 7

    # Qt Application
    app = QApplication(sys.argv)
    # QWidget
    widget = Widget(device, train, test, input_dim)
    # QMainWindow using QWidget as central widget
    window = MainWindow(widget)
    window.resize(800, 600)
    window.show()

    # Execute application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()