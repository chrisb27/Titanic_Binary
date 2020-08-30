import os
import sys
import subprocess
from PySide2.QtCore import Qt, Slot, QRunnable, QThread, QThreadPool
from PySide2.QtGui import QPainter, QFont, QPen, QBrush
from PySide2.QtWidgets import (QAction, QApplication, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QFormLayout)
from Titanicbc import Binary_Network
import torch
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from importlib import resources as res
import threading

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()  # QtCore.Slot
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)

class Widget(QWidget):
    def __init__(self, device, input_dim):
        QWidget.__init__(self)

        self.device = device
        self.input_dim = input_dim
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        with res.open_binary('Titanicbc', 'config.yaml') as fp:
            model_parameters = yaml.load(fp, Loader=yaml.Loader)

        hidden_dim_current = model_parameters['Binary_Network']['initialisations']['hidden_dim']
        learning_rate_current = model_parameters['Binary_Network']['optimiser']['learning_rate']
        num_epochs_current = model_parameters['Binary_Network']['num_epochs']
        weight_init_current = model_parameters['Binary_Network']['initialisations']['weight_init']
        weight_decay_current = model_parameters['Binary_Network']['optimiser']['weight_decay']

        # layout

        ## Read in current values from config.yaml as the default values in QForm
        self.layout = QFormLayout()

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

        self.confirm = QPushButton("Confirm network configuration and train")
        self.predict = QPushButton("Predict using last trained model")
        self.plot_loss = QPushButton("Plot loss of last trained model")
        self.output = QPushButton("Open output.csv")
        self.quit = QPushButton("Quit")

        self.layout.addRow(self.num_epochs_label, self.num_epochs)
        self.layout.addRow(self.learning_rate_label, self.learning_rate)
        self.layout.addRow(self.weight_decay_label, self.weight_decay)
        self.layout.addRow(self.hidden_dim_label, self.hidden_dim)
        self.layout.addRow(self.weight_init_label, self.weight_init)
        self.layout.addWidget(self.confirm)
        self.layout.addWidget(self.predict)
        self.layout.addWidget(self.plot_loss)
        self.layout.addWidget(self.output)
        self.layout.addWidget(self.quit)

        # Set the layout to the QWidget
        self.setLayout(self.layout)

        # Signals and Slots
        self.confirm.clicked.connect(self.confirm_thread)
        self.predict.clicked.connect(self.op_predict)
        self.plot_loss.clicked.connect(self.plot)
        self.output.clicked.connect(self.open_output)
        self.quit.clicked.connect(self.quit_application)


        ## Execution here

    def confirm_thread(self):
        worker = Worker(self.confirm_configuration)
        self.threadpool.start(worker)

    @Slot()
    def confirm_configuration(self):

        train_num_epochs = self.num_epochs.text()
        train_learning_rate = self.learning_rate.text()
        train_weight_decay = self.weight_decay.text()
        train_weight_init = self.weight_init.text()
        train_hidden_dim = self.hidden_dim.text()

        ## read about passing values out of here into
        print("Network Configuration")
        print("Number of Epochs: {}".format(train_num_epochs))
        print("Learning Rate: {}".format(train_learning_rate))
        print("Weight Decay: {}".format(train_weight_decay))
        print("Weight Initialisation: {}".format(train_weight_init))
        print("Hidden Layers Dimension: {}".format(train_hidden_dim))

        with res.open_binary('Titanicbc', 'config.yaml') as fp:
            model_parameters = yaml.load(fp, Loader=yaml.Loader)

        model_parameters['Binary_Network']['initialisations']['hidden_dim'] = int(train_hidden_dim)
        model_parameters['Binary_Network']['optimiser']['learning_rate'] = float(train_learning_rate)
        model_parameters['Binary_Network']['num_epochs'] = int(train_num_epochs)
        model_parameters['Binary_Network']['initialisations']['weight_init'] = str(train_weight_init) ## Read in Binary_Network
        model_parameters['Binary_Network']['optimiser']['weight_decay'] = float(train_weight_decay)

        ## write out parameters

        with res.path('Titanicbc', 'config.yaml') as cf:
            path = cf

        with open(path, 'w') as outfile:
            yaml.dump(model_parameters, outfile, default_flow_style=False)

        ## Read in package resources

        with res.open_binary('Titanicbc', 'train.csv') as train:
            train = pd.read_csv(train)

        with res.open_binary('Titanicbc', 'test.csv') as test:
            test = pd.read_csv(test)

        with res.path('Titanicbc', 'trained_model.pth') as m:
            model_path = m

        # All params are coming through as a string
        self.running_loss, model= Binary_Network.train_new_model(train, self.input_dim, train_hidden_dim, model_path, train_learning_rate,
                                               train_num_epochs, train_weight_decay)
        model.to(self.device)
        Binary_Network.predict(model, test)


    @Slot()
    def op_predict(self):

        with res.open_binary('Titanicbc', 'config.yaml') as fp:
            model_parameters = yaml.load(fp, Loader=yaml.Loader)

        with res.open_binary('Titanicbc', 'test.csv') as test:
            test_predict = pd.read_csv(test)

        prev_hidden_dim = model_parameters['Binary_Network']['initialisations']['hidden_dim']

        with res.path('Titanicbc', 'trained_model.pth') as m:
            model_path = m

        model = Binary_Network.Binary_Network(self.input_dim, prev_hidden_dim)
        model = Binary_Network.load_models(model_path, model).to(self.device)
        Binary_Network.predict(model, test_predict)

    @Slot()
    def plot(self):
        try:
            plt.plot(self.running_loss)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.show()
        except:
            print("Must train a new model before plotting loss")

    @Slot()
    def open_output(self):
        with res.path('Titanicbc', 'output.csv') as op:
            path = op
        output = pd.read_csv(path)
        print(output)
        try:
            os.startfile(path)
        except:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, path])


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
    input_dim = 7

    # Qt Application
    app = QApplication(sys.argv)
    # QWidget
    widget = Widget(device, input_dim)
    # QMainWindow using QWidget as central widget
    window = MainWindow(widget)
    window.resize(800, 600)
    window.show()

    # Execute application
    sys.exit(app.exec_())

if __name__ == "__main__":
    threadpool = QThreadPool()
    worker = Worker(main())
    threadpool.start(worker)