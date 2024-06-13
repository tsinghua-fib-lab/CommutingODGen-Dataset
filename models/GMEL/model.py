import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        in_dim = 131
        num_hidden = 64
        out_dim = 64

        self.num_layers = 3
        self.num_heads = 6
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        self.gat_layers.append(GATConv(in_dim, num_hidden, num_heads=self.num_heads, allow_zero_in_degree=True, activation=self.activation))

        for _ in range(1, self.num_layers):
            self.gat_layers.append(
                GATConv(num_hidden * self.num_heads, num_hidden, num_heads=self.num_heads, allow_zero_in_degree=True, activation=self.activation))

        self.gat_layers.append(GATConv(num_hidden * self.num_heads, out_dim, num_heads=self.num_heads, allow_zero_in_degree=True, activation=None))

    def forward(self, g, nfeat):
        h = nfeat
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        
        # output projection
        embeddings = self.gat_layers[-1](g, h).mean(1)

        return embeddings
    

class GMEL(nn.Module):
    def __init__(self):
        super(GMEL, self).__init__()

        self.gat_in = GAT()
        self.gat_out = GAT()

        self.linear_in = nn.Linear(64, 1)
        self.linear_out = nn.Linear(64, 1)
        self.bilinear = nn.Bilinear(64, 64, 1)

    def forward(self, g, nfeat):
        h_in = self.gat_in(g, nfeat)
        flow_in = self.linear_in(h_in)

        h_out = self.gat_out(g, nfeat)
        flow_out = self.linear_out(h_out)

        flow = self.bilinear(h_in, h_out)

        return flow_in, flow_out, flow, h_in, h_out