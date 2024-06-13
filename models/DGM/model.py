import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepGravity(nn.Module):
    def __init__(self):
        super(DeepGravity, self).__init__()

        hiddim = 256 # 256
        layers = 15 # 15

        self.linear_in = nn.Linear(263, hiddim)
        self.linears = nn.ModuleList(
            [nn.Linear(hiddim, hiddim) for i in range(layers)]
        )
        self.linear_out = nn.Linear(hiddim, 1)

    def forward(self, input):
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x)) + x
        x = torch.tanh(self.linear_out(x))
        return x
    

class OD_normer():
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_

    def normalize(self, x):
        """Scale a value or array of values to the range [-1, 1]."""
        return 2 * ((x - self.min_) / (self.max_ - self.min_)) - 1

    def renormalize(self, x):
        return ((x + 1) / 2) * (self.max_ - self.min_) + self.min_