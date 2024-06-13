import torch
import torch.nn as nn

class GRAVITY(nn.Module):
    def __init__(self):
        super(GRAVITY, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.G = nn.Parameter(torch.tensor([torch.randn(1)]))

    def forward(self, x):
        x = x + 1e-10
        logy = self.alpha * torch.log(x[:, 0]) + self.beta * torch.log(x[:, 1]) + x[:, 2] * torch.log(self.gamma**2)
        y = self.G * torch.exp(logy)
        return y