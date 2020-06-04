"""
    Model for the classification
"""
import random

import numpy as np
import torch
import torch.nn as nn



class ConvNN(nn.Module):
    def __init__(self, input_channels, image_size, output_size):
        super(ConvNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layerfc = nn.Sequential(
            nn.Linear(image_size//8 * image_size//8 * 8, output_size),
            nn.Softmax())
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0),-1)
        out = self.layerfc(out)
        return out

if __name__ == "__main__":
    image1 = torch.randn(3,512,512).unsqueeze(0)
    cnn = ConvNN(3,512,10)
    print(cnn(image1))