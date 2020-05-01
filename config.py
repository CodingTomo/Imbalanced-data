import pandas as pd
import sys


class Loader:
    def __init__(self):
        self.data = pd.read_csv("data/creditcard.csv")
        self.sample = 1

    def shuffle_sample(self):
        if self.sample == 1:
            self.data = self.data.sample(frac=0.2, replace=False, random_state=123).reset_index(drop=True)
            print('Sampling executed')
        else:
            print('Sampling not executed')


class Params:
    decision_metric: str
    label: str
    decision_threshold: float

    def __init__(self):
        self.label = "Class"
        self.decision_metric = "mean_test_roc_auc_score"
        self.decision_threshold = 0.8
        self.torch_input_dim = 30
        self.torch_hidden_dim = [25, 20, 15, 20, 25]
        self.torch_output_dim = 30
        self.torch_epochs = 20
        self.torch_lr = 0.001
        self.torch_batch = 256*2
        self.torch_id = 5


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
