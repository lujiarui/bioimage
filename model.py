"""
    Model for the classification
"""

from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Utilized device as [{}]".format(device))


# Hyper-parameters for nn
input_size = 256*256*3
hidden_size = 1000
output_size = 10
batch_size = 50
learning_rate = 10e-4

# Fully connected neural network with 1 hidden layer
# Prototype of NN
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.outLayer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.outLayer(out)
        out = self.softmax(out)
        return out

