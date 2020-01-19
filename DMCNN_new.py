import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary


class DMCNN(torch.nn.Module):

    def __init__(self):
        super(DMCNN, self).__init__()

        self.feature_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=9),
            torch.nn.ReLU()
        )

        self.mapping_layer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.ReLU()
        )

        self.reconstruction_layer = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, kernel_size=5),
            torch.nn.ReLU()
        )

    def forward(self, x):
        out = self.feature_layer(x)
        out = self.mapping_layer(out)
        out = self.reconstruction_layer(out)
        return out
