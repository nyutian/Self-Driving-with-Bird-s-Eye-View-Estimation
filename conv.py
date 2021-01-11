from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 3, kernel_size = (3, 3), padding=1)
    def forward(self, x):
        return self.conv(x)
