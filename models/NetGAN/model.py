import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import weight_norm

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


class GraphConstructor(nn.Module):

    def __init__(self):
        super(GraphConstructor, self).__init__()

        self.gnn = GAT()
        self.linear_predictor = nn.Linear(64 * 2 + 1, 1)


    def Graph_embedding(self, g, nfeat):
        node_embedding = self.gnn(g, nfeat)
        return node_embedding

    def linear_prediction(self, embeddings, distance):
        embeddings1 = embeddings.unsqueeze(dim=0).repeat(embeddings.size(0), 1, 1)
        embeddings2 = embeddings.unsqueeze(dim=1).repeat(1, embeddings.size(0), 1)
        dis = distance.unsqueeze(dim=2)

        pair_emb = torch.cat((embeddings1, embeddings2, dis), dim=2)
        OD = torch.tanh(self.linear_predictor(pair_emb).squeeze())
        # for i in range(OD.size(0)):
        #     OD[i, i] = 0
        return OD

    def forward(self, g, nfeats, dis):
        nemb = self.Graph_embedding(g, nfeats)
        adjacency = self.linear_prediction(nemb, dis)
        return adjacency
    

'''
Conditional Wasserstain GAN
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.g_constructor = GraphConstructor()
        self.threshold = nn.Threshold(threshold = 1, value = 0)

    def generate_OD_net(self, g, region_attributes, distance):
        self.OD_net = self.g_constructor(g, region_attributes, distance)
        self.adjacency = self.threshold(self.OD_net)
        self.logp = self.OD_net / (self.OD_net.sum(1).unsqueeze(dim=1) + 1e-10)
        self.logp = torch.log(self.logp + 1e-10)
        return self.OD_net, self.adjacency, self.logp

    def sample_a_neighbor(self, node):
        o_idx = torch.argmax(node)

        node = node.unsqueeze(dim=1)[o_idx].repeat(node.size(0))
        neighbors = node * self.logp[o_idx]
        next_node = F.gumbel_softmax(neighbors, hard=True)
        return next_node

    def one_hot_to_nodefeat(self, node, all_feats):
        node_idx = torch.argmax(node)

        feats = all_feats[:,:-64]
        node = node.unsqueeze(dim=1)[node_idx].repeat(feats.size(1))
        feat = feats[node_idx] * node
        return feat

    def sample_one_random_walk(self, region_attributes, distance):
        node_seq = []
        feat_seq = []
        edge_seq = []
        dis_seq = []

        init_node = torch.randint(low = 0, high = self.adjacency.size(0), size = (1,))[0].to(torch.device('cuda'))
        init_node = F.one_hot(init_node, num_classes = self.adjacency.size(0))

        node_seq.append(init_node)
        feat_seq.append(self.one_hot_to_nodefeat(init_node, region_attributes))

        for i in range(200-1):

            next_node = self.sample_a_neighbor(node_seq[-1])
            feat = self.one_hot_to_nodefeat(next_node, region_attributes)
            flow_between = self.adjacency[torch.argmax(node_seq[-1]), torch.argmax(next_node)]
            dis_between = distance[torch.argmax(node_seq[-1]), torch.argmax(next_node)]

            node_seq.append(next_node)
            feat_seq.append(feat)
            edge_seq.append(flow_between)
            dis_seq.append(dis_between)

        # node_seq = torch.stack(node_seq[1:])
        feat_seq = torch.stack(feat_seq[1:])
        edge_seq = torch.stack(edge_seq).view([-1, 1])
        dis_seq = torch.stack(dis_seq).view([-1, 1])
        # seq = torch.cat((node_seq, edge_seq, dis_seq), dim=1) # node_seq, feat_seq, edge_seq, dis_seq
        seq = edge_seq
        return seq


    def sample_generated_batch(self, g, region_attributes, distance, batch_size):
        self.generate_OD_net(g, region_attributes, distance)

        start_time = time.time()
        batch = []
        for i in range(batch_size):
            batch.append(sample_one_random_walk(self.adjacency, self.logp))
        batch = torch.stack(batch)

        return batch
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_in = nn.Linear(1, 64)
        self.tcn = TemporalConvNet(num_inputs = 64, num_channels = [64]*7)
        self.fc_pre = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc_in(x).transpose(1, 2)
        x = self.tcn(x)
        pre = self.fc_pre(x[:, :, -1]).squeeze()

        return pre


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.05):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def sample_one_random_walk(adjacency, trans_prob):
    node_seq = []
    edge_seq = []

    init_node = torch.randint(low = 0, high = adjacency.size(0), size = (1,))[0].to(torch.device('cuda'))
    init_node = F.one_hot(init_node, num_classes = adjacency.size(0))

    node_seq.append(init_node)

    for _ in range(200-1):
        next_node = sample_a_neighbor(trans_prob, node_seq[-1])
        flow_between = adjacency[torch.argmax(node_seq[-1]), torch.argmax(next_node)]

        edge_seq.append(flow_between)

    edge_seq = torch.stack(edge_seq).view([-1, 1])
    return edge_seq


def one_hot_to_nodefeat(node, all_feats):
    node_idx = torch.argmax(node)

    feats = all_feats[:, :-64]
    node = node.unsqueeze(dim=1)[node_idx].repeat(feats.size(1))
    feat = feats[node_idx] * node
    return feat


def sample_a_neighbor(trans_prob, node):
    o_idx = torch.argmax(node)
    node = node.unsqueeze(dim=1)[o_idx].repeat(node.size(0))
    neighbors = node * trans_prob[o_idx]
    next_node = F.gumbel_softmax(neighbors, tau = 1, hard=True)
    return next_node


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1))).to(torch.device('cuda'))
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.FloatTensor(real_samples.shape[0]).fill_(1.0).to(torch.device('cuda'))
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty