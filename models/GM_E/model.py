import torch
import torch.nn as nn

class GRAVITY(nn.Module):
    def __init__(self):
        super(GRAVITY, self).__init__()
        self.linear = nn.Linear(3, 1)
        self.G = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = x + 1e-10
        x = torch.log(x)
        logy = self.linear(x)
        y = self.G * torch.exp(logy)
        return y