import torch
import torch.nn as nn


class RegTokenPooling(nn.Module):
    def __init__(self):
        super(RegTokenPooling, self).__init__()

    def forward(self, x, attention_mask=None):
        x = x[:, 1:7, :]
        return x
