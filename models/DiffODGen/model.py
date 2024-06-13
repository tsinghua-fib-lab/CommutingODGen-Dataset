from copy import deepcopy
import os

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tqdm import tqdm
from dgl.nn import GATConv
from utils.tool import get_named_beta_schedule, timestep_embedding, node_feat_from_adj, MarginalUniformTransition, DiscreteUniformTransition, compute_over0_posterior_distribution



class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        xdim = config["GTN_x_hiddim"]
        self.n_head = config["GTN_n_head"]
        self.df = int(xdim / self.n_head)
        
        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 3:
            x = x.reshape([-1, x.shape[-1]])

        Q = self.q(x)
        K = self.k(x)
        V = V.reshape((V.size(0), self.n_head, self.df))

        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))

        Q = Q.unsqueeze(1)                             # (1, n, n_head, df)
        K = K.unsqueeze(0)                             # (n, 1, n head, df)

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603

        attn = F.softmax(Y, dim=1)

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape([*shape[:-1], -1])

        return attn


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["GTN_x_hiddim"]
        self.n_head = config["GTN_n_head"]
        self.df = int(xdim / self.n_head)

        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

    def forward(self, x, c):
        shape = x.shape
        if len(shape) == 3:
            x = x.reshape([-1, x.shape[-1]])
            c = c.reshape([-1, c.shape[-1]])

        Q, K, V = self.q(x), self.k(c), self.v(c)

        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        V = V.reshape((V.size(0), self.n_head, self.df))
        

        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)
        V = V.unsqueeze(0)                             # (1, n, n_head, df)

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603

        attn = F.softmax(Y, dim=1)

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V


class NodeEdgeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["GTN_x_hiddim"]
        edim = config["GTN_e_hiddim"]
        self.n_head = config["GTN_n_head"]
        self.df = int(xdim / self.n_head)
        assert xdim % self.n_head == 0, f"dx: {dx} -- nhead: {self.n_head}"

        # Attention
        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

        # FiLM E to X
        self.e_add = nn.Linear(edim, xdim)
        self.e_mul = nn.Linear(edim, xdim)

        # Output layers
        self.x_out = nn.Linear(xdim, xdim)
        self.e_out = nn.Linear(xdim, edim)

    def forward(self, x, e, adj):
        # Map X to keys and queries
        Q = self.q(x)
        K = self.k(x) # 2639
        
        # Reshape to (n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        
        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)

        # Compute unnormalized attentions.
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603
        
        E1 = self.e_mul(e)                             # (n, n, dx)
        E1 = E1.reshape((e.size(0), e.size(1), self.n_head, self.df)) # 3603
        
        E2 = self.e_add(e)
        E2 = E2.reshape((e.size(0), e.size(1), self.n_head, self.df)) # 4085
        
        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (n, n, n_head, df) # 5531
        
        # Output E
        newE = Y.flatten(start_dim=2)
        newE = self.e_out(newE) # 5531
        
        # Compute attentions. attn is still (n, n, n_head, df)
        attn = F.softmax(Y, dim=1) # * adj.unsqueeze(-1).unsqueeze(-1) # 6495
        
        # Map X to values
        V = self.v(x)                        # n, dx
        V = V.reshape((V.size(0), self.n_head, self.df))
        V = V.unsqueeze(0)                   # (1, n, n_head, df) # 6495
        
        # Compute weighted values
        weighted_V = attn * V                # (n, n, n_head, df)
        weighted_V = weighted_V.sum(dim=1)   # (n, n_head, df) # 7041
        
        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=1)            # (n, dx)

        # Output X
        newX = self.x_out(weighted_V) # 7043
        
        return newX, newE


class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim, edim = config["GTN_x_hiddim"], config["GTN_e_hiddim"]
        dim_ffX, dim_ffE = config["GTN_dim_ffX"], config["GTN_dim_ffE"]

        self.self_attn = NodeEdgeBlock(config)

        self.linX1 = nn.Linear(xdim, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, xdim)
        self.normX1 = nn.LayerNorm(xdim)
        self.normX2 = nn.LayerNorm(xdim)
        self.dropoutX1 = nn.Dropout(config["GTN_dropout"])
        self.dropoutX2 = nn.Dropout(config["GTN_dropout"])
        self.dropoutX3 = nn.Dropout(config["GTN_dropout"])

        self.linE1 = nn.Linear(edim, dim_ffE)
        self.linE2 = nn.Linear(edim, dim_ffE)
        self.normE1 = nn.LayerNorm(edim)
        self.normE2 = nn.LayerNorm(edim)
        self.dropoutE1 = nn.Dropout(config["GTN_dropout"])
        self.dropoutE2 = nn.Dropout(config["GTN_dropout"])
        self.dropoutE3 = nn.Dropout(config["GTN_dropout"])


    def forward(self, x, e, adj):
        newX, newE = self.self_attn(x, e, adj) # 7043
        
        newX_d = self.dropoutX1(newX)
        x = self.normX1(x + newX_d)

        newE_d = self.dropoutE1(newE)
        e = self.normE1(e + newE_d) # 7043

        ff_outputX = self.linX2(self.dropoutX2(F.relu(self.linX1(x), inplace=True)))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(x + ff_outputX) # 7043
        
        ff_outputE = self.linE2(self.dropoutE2(F.relu(self.linE1(e), inplace=True)))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(e + ff_outputE) # 8489
        
        return X, E


class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        x_indim = config["GTN_x_indim"]
        e_indim = config["GTN_e_indim"]
        x_hiddim = config["GTN_x_hiddim"]
        e_hiddim = config["GTN_e_hiddim"]
        x_outdim = config["GTN_x_outdim"]
        e_outdim = config["GTN_e_outdim"]

        self.x_outdim = x_outdim
        self.e_outdim = e_outdim

        self.xin = nn.Sequential(
            nn.Linear(x_indim, x_hiddim), nn.ReLU(inplace=True),
            nn.Linear(x_hiddim, x_hiddim, nn.ReLU(inplace=True))
        ).cuda()
        self.ein = nn.Sequential(
            nn.Linear(e_indim, e_hiddim), nn.ReLU(inplace=True),
            nn.Linear(e_hiddim, e_hiddim, nn.ReLU(inplace=True))
        ).cuda()

        self.GTL1 = GraphTransformerLayer(config).cuda()
        self.GTL2 = GraphTransformerLayer(config).cuda()
        self.GTL3 = GraphTransformerLayer(config).cuda()

        self.linear_x1 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_x2 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_x3 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        # self.linear_x4 = nn.Sequential(
        #     nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        # ) .cuda()

        self.linear_e1 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_e2 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_e3 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        # self.linear_e4 = nn.Sequential(
        #     nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        # ) .cuda()

        self.cross_attn_x1 = CrossAttention(config).cuda()
        self.cross_attn_x2 = CrossAttention(config).cuda()
        self.cross_attn_x3 = CrossAttention(config).cuda()
        # self.cross_attn_x4 = CrossAttention(config).cuda()
        
        self.eout = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True),
            nn.Linear(e_hiddim, e_outdim)
        ).cuda()

    def forward(self, x, e, adj, emb):
        n_emb, e_emb = emb
        
        x = self.xin(x.to(self.xin[0].weight.device))
        e = self.ein(e.to(self.xin[0].weight.device))
        
        x_h, e_h = x, e
        x, e = x[..., :self.x_outdim], e[..., :self.e_outdim]
        
        # layer1
        n_emb, e_emb = n_emb.to(self.GTL1.linX1.weight.device), e_emb.to(self.GTL1.linX1.weight.device)
        n_emb, e_emb = self.linear_x1(n_emb), self.linear_e1(e_emb)
        x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = n_emb, e_h + e_emb
        x_h, e_h = self.cross_attn_x1(x_h, n_emb), e_h + e_emb
        # x_h, e_h = x_h + n_emb, e_h + e_emb
        adj = adj.to(x_h.device)
        x_h, e_h = self.GTL1(x_h, e_h, adj)
        
        # layer2
        n_emb, e_emb = n_emb.to(self.GTL2.linX1.weight.device), e_emb.to(self.GTL2.linX1.weight.device)
        n_emb, e_emb = self.linear_x2(n_emb), self.linear_e2(e_emb)
        x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = n_emb, e_h + e_emb
        x_h, e_h = self.cross_attn_x2(x_h, n_emb), e_h + e_emb
        # x_h, e_h = x_h + n_emb, e_h + e_emb
        adj = adj.to(x_h.device)
        x_h, e_h = self.GTL2(x_h, e_h, adj)

        # layer3
        n_emb, e_emb = n_emb.to(self.GTL3.linX1.weight.device), e_emb.to(self.GTL3.linX1.weight.device)
        n_emb, e_emb = self.linear_x3(n_emb), self.linear_e3(e_emb)
        x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = n_emb, e_h + e_emb
        x_h, e_h = self.cross_attn_x3(x_h, n_emb), e_h + e_emb
        # x_h, e_h = x_h + n_emb, e_h + e_emb
        adj = adj.to(x_h.device)
        x_h, e_h = self.GTL3(x_h, e_h, adj)

        # # layer4
        # n_emb, e_emb = n_emb.to(self.GTL4.linX1.weight.device), e_emb.to(self.GTL4.linX1.weight.device)
        # n_emb, e_emb = self.linear_x4(n_emb), self.linear_e4(e_emb)
        # x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = self.cross_attn_x4(x_h, n_emb), e_h + e_emb
        # # x_h, e_h = x_h + n_emb, e_h + e_emb
        # adj = adj.to(x_h.device)
        # x_h, e_h = self.GTL4(x_h, e_h, adj)
        
        e_out = self.eout(e_h.to(self.eout[0].weight.device))
        e_out = e_out + e.to(e_out.device)
        
        return e_out


class topo_CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["T_GTN_x_hiddim"]
        self.n_head = config["GTN_n_head"]
        self.df = int(xdim / self.n_head)

        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

    def forward(self, x, c):
        shape = x.shape
        if len(shape) == 3:
            x = x.reshape([-1, x.shape[-1]])
            c = c.reshape([-1, c.shape[-1]])

        Q, K, V = self.q(x), self.k(c), self.v(c)

        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        V = V.reshape((V.size(0), self.n_head, self.df))
        

        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)
        V = V.unsqueeze(0)                             # (1, n, n_head, df)

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603

        attn = F.softmax(Y, dim=1)

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V


class topo_NodeEdgeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["T_GTN_x_hiddim"]
        edim = config["T_GTN_e_hiddim"]
        self.n_head = config["GTN_n_head"]
        self.df = int(xdim / self.n_head)
        assert xdim % self.n_head == 0, f"dx: {dx} -- nhead: {self.n_head}"

        # Attention
        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

        # FiLM E to X
        self.e_add = nn.Linear(edim, xdim)
        self.e_mul = nn.Linear(edim, xdim)

        # Output layers
        self.x_out = nn.Linear(xdim, xdim)
        self.e_out = nn.Linear(xdim, edim)

    def forward(self, x, e, adj):
        # Map X to keys and queries
        Q = self.q(x)
        K = self.k(x) # 2639
        
        # Reshape to (n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        
        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)

        # Compute unnormalized attentions.
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603
        
        E1 = self.e_mul(e)                             # (n, n, dx)
        E1 = E1.reshape((e.size(0), e.size(1), self.n_head, self.df)) # 3603
        
        E2 = self.e_add(e)
        E2 = E2.reshape((e.size(0), e.size(1), self.n_head, self.df)) # 4085
        
        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (n, n, n_head, df) # 5531
        
        # Output E
        newE = Y.flatten(start_dim=2)
        newE = self.e_out(newE) # 5531
        
        # Compute attentions. attn is still (n, n, n_head, df)
        attn = F.softmax(Y, dim=1) # * adj.unsqueeze(-1).unsqueeze(-1) # 6495
        
        # Map X to values
        V = self.v(x)                        # n, dx
        V = V.reshape((V.size(0), self.n_head, self.df))
        V = V.unsqueeze(0)                   # (1, n, n_head, df) # 6495
        
        # Compute weighted values
        weighted_V = attn * V                # (n, n, n_head, df)
        weighted_V = weighted_V.sum(dim=1)   # (n, n_head, df) # 7041
        
        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=1)            # (n, dx)

        # Output X
        newX = self.x_out(weighted_V) # 7043
        
        return newX, newE


class topo_GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim, edim = config["T_GTN_x_hiddim"], config["T_GTN_e_hiddim"]
        dim_ffX, dim_ffE = config["T_GTN_dim_ffX"], config["T_GTN_dim_ffE"]

        self.self_attn = topo_NodeEdgeBlock(config)

        self.linX1 = nn.Linear(xdim, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, xdim)
        self.normX1 = nn.LayerNorm(xdim)
        self.normX2 = nn.LayerNorm(xdim)
        self.dropoutX1 = nn.Dropout(config["GTN_dropout"])
        self.dropoutX2 = nn.Dropout(config["GTN_dropout"])
        self.dropoutX3 = nn.Dropout(config["GTN_dropout"])

        self.linE1 = nn.Linear(edim, dim_ffE)
        self.linE2 = nn.Linear(edim, dim_ffE)
        self.normE1 = nn.LayerNorm(edim)
        self.normE2 = nn.LayerNorm(edim)
        self.dropoutE1 = nn.Dropout(config["GTN_dropout"])
        self.dropoutE2 = nn.Dropout(config["GTN_dropout"])
        self.dropoutE3 = nn.Dropout(config["GTN_dropout"])


    def forward(self, x, e, adj):
        newX, newE = self.self_attn(x, e, adj) # 7043
        
        newX_d = self.dropoutX1(newX)
        x = self.normX1(x + newX_d)

        newE_d = self.dropoutE1(newE)
        e = self.normE1(e + newE_d) # 7043

        ff_outputX = self.linX2(self.dropoutX2(F.relu(self.linX1(x), inplace=True)))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(x + ff_outputX) # 7043
        
        ff_outputE = self.linE2(self.dropoutE2(F.relu(self.linE1(e), inplace=True)))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(e + ff_outputE) # 8489
        
        return X, E


class topo_GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        x_indim = config["T_GTN_x_indim"]
        e_indim = config["T_GTN_e_indim"]
        x_hiddim = config["T_GTN_x_hiddim"]
        e_hiddim = config["T_GTN_e_hiddim"]
        x_outdim = config["T_GTN_x_outdim"]
        e_outdim = config["T_GTN_e_outdim"]

        self.x_outdim = x_outdim
        self.e_outdim = e_outdim

        self.xin = nn.Sequential(
            nn.Linear(x_indim, x_hiddim), nn.ReLU(inplace=True)
        ).cuda()
        self.ein = nn.Sequential(
            nn.Linear(e_indim, e_hiddim), nn.ReLU(inplace=True)
        ).cuda()

        self.GTL1 = topo_GraphTransformerLayer(config).cuda()
        self.GTL2 = topo_GraphTransformerLayer(config).cuda()
        self.GTL3 = topo_GraphTransformerLayer(config).cuda()
        self.GTL4 = topo_GraphTransformerLayer(config).cuda()

        self.linear_x1 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_x2 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_x3 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_x4 = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True)
        ) .cuda()

        self.linear_e1 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_e2 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_e3 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()
        self.linear_e4 = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True)
        ) .cuda()

        self.cross_attn_x1 = topo_CrossAttention(config).cuda()
        self.cross_attn_x2 = topo_CrossAttention(config).cuda()
        self.cross_attn_x3 = topo_CrossAttention(config).cuda()
        self.cross_attn_x4 = topo_CrossAttention(config).cuda()

        self.eout = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True),
            nn.Linear(e_hiddim, e_outdim)
        ).cuda()
        

    def forward(self, x, e, adj, emb):
        n_emb, e_emb = emb

        x = self.xin(x.to(self.xin[0].weight.device))
        e = self.ein(e.to(self.xin[0].weight.device))
        
        x_h, e_h = x, e
        # x, e = x[..., :self.x_outdim], e[..., :self.e_outdim]
        e = e[..., :self.e_outdim]
        
        # layer1
        n_emb, e_emb = n_emb.to(self.GTL1.linX1.weight.device), e_emb.to(self.GTL1.linX1.weight.device)
        n_emb, e_emb = self.linear_x1(n_emb), self.linear_e1(e_emb)
        x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = n_emb, e_h + e_emb
        # x_h, e_h = x_h + n_emb, e_h + e_emb
        x_h, e_h = self.cross_attn_x1(x_h, n_emb), e_h + e_emb
        adj = adj.to(x_h.device)
        x_h, e_h = self.GTL1(x_h, e_h, adj)
        
        # layer2
        n_emb, e_emb = n_emb.to(self.GTL2.linX1.weight.device), e_emb.to(self.GTL2.linX1.weight.device)
        n_emb, e_emb = self.linear_x2(n_emb), self.linear_e2(e_emb)
        x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # x_h, e_h = n_emb, e_h + e_emb
        # x_h, e_h = x_h + n_emb, e_h + e_emb
        x_h, e_h = self.cross_attn_x2(x_h, n_emb), e_h + e_emb
        adj = adj.to(x_h.device)
        x_h, e_h = self.GTL2(x_h, e_h, adj)

        # # layer3
        # n_emb, e_emb = n_emb.to(self.GTL3.linX1.weight.device), e_emb.to(self.GTL3.linX1.weight.device)
        # n_emb, e_emb = self.linear_x3(n_emb), self.linear_e3(e_emb)
        # x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # # x_h, e_h = x_h + n_emb, e_h + e_emb
        # x_h, e_h = self.cross_attn_x3(x_h, n_emb), e_h + e_emb
        # adj = adj.to(x_h.device)
        # x_h, e_h = self.GTL3(x_h, e_h, adj)

        # # layer4
        # n_emb, e_emb = n_emb.to(self.GTL4.linX1.weight.device), e_emb.to(self.GTL4.linX1.weight.device)
        # n_emb, e_emb = self.linear_x4(n_emb), self.linear_e4(e_emb)
        # x_h, e_h = x_h.to(n_emb.device), e_h.to(e_emb.device)
        # # x_h, e_h = x_h + n_emb, e_h + e_emb
        # x_h, e_h = self.cross_attn_x4(x_h, n_emb), e_h + e_emb
        # adj = adj.to(x_h.device)
        # x_h, e_h = self.GTL4(x_h, e_h, adj)
        
        e_out = self.eout(e_h.to(self.eout[0].weight.device))

        e = e.to(e_out.device)
        e_out = e_out + e
        
        return e_out


class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.num_layers = config["GAT_layers"] -1
        num_in = config["GAT_indim"]
        num_hidden = config["GAT_hiddim"]
        num_out = config["GAT_outdim"]
        heads = [config["GAT_heads"]] * 6
        feat_drop = config["GAT_feat_drop"]
        attn_drop = config["GAT_attn_drop"]
        negative_slope = config["GAT_negative_slope"]
        residual = config["GAT_residual"]
        activation = F.elu

        self.gat_layers = nn.ModuleList()
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(num_in, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, activation))

        # hidden layers
        for l in range(0, self.num_layers-1):
            # due to multi-head, the num_in = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, activation))

        if self.num_layers >= 1:
            # output projection
            self.gat_layers.append(
                GATConv(num_hidden * heads[-2], num_out, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, g):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        
        if self.num_layers >= 1:
            # output projection
            h = self.gat_layers[-1](g, h).mean(1)

        return h


class disProj(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_in = nn.Linear(1, config["disProj_hiddim"])

        self.linears = nn.ModuleList(
            [nn.Linear(config["disProj_hiddim"], config["disProj_hiddim"]) for i in range(config["disProj_layers"])]
        )

    def forward(self, dis):
        dis = dis.unsqueeze(-1)

        dis = self.linear_in(dis)
        mid = dis
        for layer in self.linears:
            mid = F.silu(layer(mid), inplace=True)
        out = mid
        return out


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config["GAT_outdim"] * 2 + config["disProj_hiddim"], config["GAT_outdim"] - 1)

    def forward(self, o_emb, d_emb, dis):
        o_emb = o_emb.unsqueeze(1).expand(o_emb.size(0), o_emb.size(0), o_emb.size(1))
        d_emb = d_emb.unsqueeze(0).expand(d_emb.size(0), d_emb.size(0), d_emb.size(1))
        
        hid = torch.concat([o_emb, d_emb, dis], dim=-1)
        flow = F.silu(self.fc(hid))
        # flow = torch.tanh(self.fc(hid))
        return flow


class PreNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.GAT_o = GAT(config).cuda()
        self.GAT_d = GAT(config).cuda()

        self.dis_proj = disProj(config).cuda()

        self.flow_predictor = Predictor(config).cuda()

        self.n_emb_linear = nn.Linear(config["GAT_outdim"] * 2, config["GAT_outdim"]).cuda()

    def forward(self, attr, dis, g):
        attr = attr.to(self.GAT_o.gat_layers[0].fc.weight.device)
        g = g.to(attr.device)

        o_emb = self.GAT_o(attr, g)
        d_emb = self.GAT_d(attr, g)
        
        dis = self.dis_proj(dis.to(self.dis_proj.linear_in.weight.device))
        
        e_emb = self.flow_predictor(o_emb, d_emb, dis.to(o_emb.device))
        n_emb = self.n_emb_linear(torch.concat((o_emb, d_emb), dim=-1))
        return n_emb, e_emb


class flow_con_linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.linear_o = nn.Sequential(
            nn.Linear(config["GAT_indim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_outdim"])
        ).cuda()
        self.linear_d = nn.Sequential(
            nn.Linear(config["GAT_indim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_outdim"])
        ).cuda()

        self.dis_proj = disProj(config).cuda()

        self.n_emb_linear = nn.Linear(config["GAT_outdim"] * 2, config["GAT_outdim"]).cuda()
        self.flow_predictor = Predictor(config).cuda()

    def forward(self, attr, dis):
        attr = attr.to(self.linear_o[0].weight.device)
        o_emb = self.linear_o(attr)
        d_emb = self.linear_d(attr)

        dis = self.dis_proj(dis.to(self.dis_proj.linear_in.weight.device))

        n_emb = self.n_emb_linear(torch.concat((o_emb, d_emb), dim=-1))
        e_emb = self.flow_predictor(o_emb, d_emb, dis.to(o_emb.device))
        return n_emb, e_emb


class topo_con_linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.linear_o = nn.Sequential(
            nn.Linear(config["GAT_indim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_outdim"])
        ).cuda()
        self.linear_d = nn.Sequential(
            nn.Linear(config["GAT_indim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_hiddim"]), nn.ReLU(inplace=True),
            nn.Linear(config["GAT_hiddim"], config["GAT_outdim"])
        ).cuda()

        self.dis_proj = disProj(config).cuda()

        self.n_emb_linear = nn.Linear(config["GAT_outdim"] * 2, config["GAT_outdim"]).cuda()
        self.flow_predictor = Predictor(config).cuda()

    def forward(self, attr, dis):
        attr = attr.to(self.linear_o[0].weight.device)
        o_emb = self.linear_o(attr)
        d_emb = self.linear_d(attr)

        dis = self.dis_proj(dis.to(self.dis_proj.linear_in.weight.device))

        n_emb = self.n_emb_linear(torch.concat((o_emb, d_emb), dim=-1))
        e_emb = self.flow_predictor(o_emb, d_emb, dis.to(o_emb.device))
        return n_emb, e_emb


class TopoNet_MLP(nn.Module):
    """
    To be finised.
    """
    def __init__(self, config):
        super().__init__()

        self.topo_preNN = PreNN(config)

        self.conv_out = nn.Conv2d()

    def forward(self, attr, dis, g, x_t, t):
        topo_pre = self.topo_preNN(attr, dis, g)

        x_t_minus_1 = 0
        return x_t_minus_1


class TopoNet_GTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["topo_Cemb"] == "gnn":
            self.topo_preNN = PreNN(config)
        elif config["topo_Cemb"] == "mlp":
            self.topo_preNN = topo_con_linear(config)

        self.linear_nemb = nn.Linear(config["GAT_outdim"], config["T_GTN_x_hiddim"]).to(self.topo_preNN.n_emb_linear.weight.device)
        self.linear_eemb = nn.Linear(config["GAT_outdim"], config["T_GTN_e_hiddim"]).to(self.topo_preNN.n_emb_linear.weight.device)

        self.GTN = topo_GraphTransformer(config)

    def forward(self, attr, dis, g, x_t, t):
        # condition process
        if self.config["topo_Cemb"] == "gnn":
            n_emb, e_emb = self.topo_preNN(attr, dis, g)
        elif self.config["topo_Cemb"] == "mlp":
            n_emb, e_emb = self.topo_preNN(attr, dis)
        e_emb = torch.concat((e_emb, torch.ones(e_emb.size(0), e_emb.size(0), 1).to(e_emb.device)), dim=-1)

        # # if no condition
        # n_emb, e_emb = torch.zeros_like(n_emb).to(n_emb.device), torch.zeros_like(e_emb).to(e_emb.device)

        # condition and time embedding
        t = t.to(e_emb.device)
        t_emb = timestep_embedding(self.config, t, dim=n_emb.size(-1), max_period=self.config["T_topo"])
        e_emb = t_emb + e_emb
        n_emb = t_emb + n_emb
        n_emb, e_emb = self.linear_nemb(n_emb), self.linear_eemb(e_emb)
        emb = (n_emb, e_emb)

        # adj mask, for g in Graph Transformer
        adj = torch.zeros_like(x_t[:, :, 0])
        adj[g.edges()] = 1

        # prepare node feautres from x_t (generated flow edges)
        tmp_node_feat = node_feat_from_adj(x_t[:, :, 1])
        
        # Graph Transformer
        x_t_minus_1 = self.GTN(tmp_node_feat, x_t, adj, emb)

        return x_t_minus_1


class TopoDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.NN = TopoNet_GTN(config)

        betas = get_named_beta_schedule(schedule_name=config["beta_scheduler_topo"], num_diffusion_timesteps=config["T_topo"])
        betas = torch.FloatTensor(betas).cuda()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = self.num_timesteps
        self.CE_weight = torch.FloatTensor([1 - config["Topo_e_weight"], config["Topo_e_weight"]])

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Q transition matrix modeling
        self.transition_model = DiscreteUniformTransition(e_classes=config["Topo_e_classes"])

    def q_sample(self, clean_topo, t):

        alpha_t_bar = self.alphas_cumprod[t]

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)

        # Compute transition probabilities
        probE = clean_topo.to(Qtb.device) @ Qtb

        tmp_dim = probE.size(0)
        probE = probE.reshape([-1, probE.size(-1)])
        E_t = probE.multinomial(1).squeeze().reshape([tmp_dim, tmp_dim])
        E_t = F.one_hot(E_t, num_classes=self.config["Topo_e_classes"]).float()

        return E_t

    def loss(self, x_0, t, condition):
        x_t = self.q_sample(x_0, t)

        x_0_hat = self.NN_predict(condition, x_t, t)

        x_0 = x_0.reshape([-1, x_0.size(-1)]).to(x_0_hat.device)
        x_0_hat = x_0_hat.reshape([-1, x_0_hat.size(-1)])

        loss_diff = F.cross_entropy(x_0_hat, x_0, weight=self.CE_weight.to(x_0.device))

        return loss_diff

    def NN_predict(self, condition, x_t, t):
        attr, dis, g, _ = condition
        return self.NN(attr, dis, g, x_t, t)

    def digress_sample(self, x_t, t, condition):
        px_0_hat = self.NN_predict(condition, x_t, t)
        px_0_hat = F.softmax(px_0_hat, dim=-1)

        n = px_0_hat.size(0)

        # Retrieve transitions matrix
        beta_t = self.betas[t]
        alpha_s_bar = self.alphas_cumprod[t-1]
        alpha_t_bar = self.alphas_cumprod[t]
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar)
        Qt = self.transition_model.get_Qt(beta_t)
        p_s_and_t_given_0 = compute_over0_posterior_distribution(X_t=x_t.to(Qt.device), Qt=Qt, Qsb=Qsb, Qtb=Qtb) # N, d0, d_t-1

        # Dim of these two tensors: N, d0, d_t-1
        px_0_hat = px_0_hat.reshape([-1, px_0_hat.size(-1)]) # N, d0
        weighted_X = px_0_hat.unsqueeze(-1) * p_s_and_t_given_0.to(px_0_hat.device)         # n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=1)                     # n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # n, d_t-1

        x_t = F.one_hot(prob_X.multinomial(1), num_classes=self.config["Topo_e_classes"]).float().reshape([n, n, prob_X.size(-1)])
        
        return x_t
        
    def d3pm_sample(self, x_t, t, condition):
        px_0_hat = self.NN_predict(condition, x_t, t)
        px_0_hat = F.softmax(px_0_hat, dim=-1)

        n = px_0_hat.size(0)
        
        beta_t = self.betas[t]
        alpha_s_bar = self.alphas_cumprod[t-1]

        Qt = self.transition_model.get_Qt(beta_t)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar)

        Qt_T = Qt.transpose(-1, -2)

        transition_probs = x_t @ Qt_T
        prob_at_time_t_minus_1 = px_0_hat @ Qsb

        posterior = transition_probs * prob_at_time_t_minus_1
        posterior = posterior / posterior.sum(-1).unsqueeze(-1)

        x_t = F.one_hot(posterior.reshape([-1, posterior.shape[-1]]).multinomial(1), num_classes=self.config["Topo_e_classes"]).float().reshape([n, n, posterior.shape[-1]])

        return x_t

    def x_t_minus_1_sample(self, x_t, t, condition):
        # p x_t_minus_1
        if self.config["topo_predict_loss"] == 1:
            p_x_t_minus_1, _ = self.NN_predict(condition, x_t, t)
        else:
            p_x_t_minus_1 = self.NN_predict(condition, x_t, t)
        p_x_t_minus_1 = F.softmax(p_x_t_minus_1, dim=-1)

        n = p_x_t_minus_1.shape[0]

        x_t_minus_1 = F.one_hot(p_x_t_minus_1.reshape([-1, p_x_t_minus_1.shape[-1]]).multinomial(1), num_classes=self.config["Topo_e_classes"]).float().reshape([n, n, p_x_t_minus_1.shape[-1]])

        return x_t_minus_1


    @torch.no_grad()
    def sample(self, shape, condition, marginals=None):
        # 生成初始的 x_T
        if marginals == None: # 不考虑边缘分布，直接均匀分布得到噪声
            num_class_e = self.config["Topo_e_classes"]
            p_x_T = torch.ones([shape[0], shape[1], num_class_e]) / num_class_e
            x_T = F.one_hot(p_x_T.reshape([-1, num_class_e]).multinomial(1).reshape([shape[0], shape[1]]).long(), num_classes=num_class_e).float()
        else:
            pass

        # 循环依次得到之前的结果
        x_t = x_T.to(self.betas[0].device)
        ts = [x for x in reversed(range(1, self.config["T_topo"]))]
        for t in tqdm(ts):
            t = torch.LongTensor([t])
            if self.config["topo_sample_step"] == "d3pm":
                x_t = self.d3pm_sample(x_t, t, condition)
            elif self.config["topo_sample_step"] == "digress":
                x_t = self.digress_sample(x_t, t, condition)
            elif self.config["topo_sample_step"] == "x_t_minus_1":
                x_t = self.x_t_minus_1_sample(x_t, t, condition)

            # if t.item() < self.num_timesteps:
            #     if not os.path.exists("draft/check_ts/" + self.config["exp_name"]):
            #         os.mkdir("draft/check_ts/" + self.config["exp_name"])
            #     np.save("draft/check_ts/" + self.config["exp_name"] + "/" + str(int(t.item())) + ".npy", x_t.cpu().numpy())
        x_0_sample = torch.argmax(x_t, dim=-1)
        
        return x_0_sample
        
    
class FlowNet_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.flow_preNN = PreNN(config)

        self.linear_emb = nn.Linear(config["NN_outdim"], config["diffFusion_dim"]).to(config["devices"][1])

        self.linears = nn.ModuleList(
            [ nn.Linear(config["diffFusion_dim"], config["diffFusion_dim"]) ] + 
            [ nn.Linear(config["diffFusion_dim"], config["diffFusion_dim"]) for _ in range(config["diffFusion_layers"]-2) ] + 
            [ nn.Linear(config["diffFusion_dim"], 1) ]
        ).to(config["devices"][2])

    def forward(self, attr, dis, g, x_t, t):
        n_emb, e_emb = self.flow_preNN(attr, dis, g)

        t = t.cuda()
        t_emb = timestep_embedding(self.config, t, dim=self.config["NN_outdim"], max_period=self.config["T_flow"])
        emb = self.linear_emb(t_emb + e_emb).cuda()

        x_t = x_t.unsqueeze(-1).cuda()
        h = x_t
        for i in range(len(self.linears)):
            h = h + emb
            if i != len(self.linears) -1:
                h = F.silu(self.linears[i](h))
            else:
                h = self.linears[i](h)

        epsilon_t_minus_1 = h.squeeze()
        return epsilon_t_minus_1


class FlowNet_GTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.flow_preNN = PreNN(config)

        self.linear_nemb = nn.Linear(config["GAT_outdim"], config["GTN_x_hiddim"]).to(self.flow_preNN.n_emb_linear.weight.device)
        self.linear_eemb = nn.Linear(config["GAT_outdim"], config["GTN_e_hiddim"]).to(self.flow_preNN.n_emb_linear.weight.device)

        self.GTN = GraphTransformer(config)

    def forward(self, attr, dis, g, x_t, t, topo = None):
        # condition process
        n_emb, e_emb = self.flow_preNN(attr, dis, g)
        
        # topo introduced
        topo_emb = topo.unsqueeze(-1).to(e_emb.device)
        e_emb = torch.concat((e_emb, topo_emb), dim=-1)
        
        # condition and time embedding
        t = t.to(e_emb.device)
        t_emb = timestep_embedding(self.config, t, dim=n_emb.size(-1), max_period=self.config["T_flow"])
        e_emb = t_emb + e_emb
        n_emb = t_emb + n_emb
        n_emb, e_emb = self.linear_nemb(n_emb), self.linear_eemb(e_emb)
        emb = (n_emb, e_emb)
        
        # adj mask, for g in Graph Transformer
        adj = torch.zeros_like(x_t)
        adj[g.edges()] = 1
        
        # prepare node feautres from x_t (generated flow edges)
        tmp_node_feat = node_feat_from_adj(x_t)
        
        # Graph Transformer
        x_t_minus_1 = self.GTN(tmp_node_feat, x_t.unsqueeze(-1), adj, emb)
        x_t_minus_1 = x_t_minus_1.squeeze()
        
        # topo mask
        zero_flow_idx = topo == 0
        x_t_minus_1[zero_flow_idx] = 0 # zero flows are normalized to -1, the denoise should be 0.
        
        return x_t_minus_1


class FlowDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config= config

        if config["NN_type"] == "MLP":
            self.NN = FlowNet_MLP(config)
        elif config["NN_type"] == "GTN":
            self.NN = FlowNet_GTN(config)

        betas = get_named_beta_schedule(schedule_name=config["beta_scheduler_flow"], num_diffusion_timesteps=config["T_flow"])
        betas = torch.FloatTensor(betas).cuda()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = self.num_timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_0, t, noise):
        '''
        diffusion process
        '''
        if noise == None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def loss(self, x_0, t, condition):
        '''
        MSE with noise now
        '''
        epsilon_noise = torch.randn_like(x_0).to(self.betas.device)
        topo = condition[3]
        zero_flow_idx = topo == 0
        epsilon_noise[zero_flow_idx] = 0 # zero flows are normalized to -1, the denoise should be 0.

        x_t = self.q_sample(x_0, t, epsilon_noise)
        epsilon_hat = self.NN_predict(condition, x_t, t).to(epsilon_noise.device)
        return self.loss_fn(epsilon_hat, epsilon_noise)


    def NN_predict(self, condition, x_t, t):
        attr, dis, g, topo = condition
        return self.NN(attr, dis, g, x_t, t, topo)

    @property
    def loss_fn(self):
        return F.mse_loss

    def p_sample(self, x_t, t, condition):
        '''
        reverse denoise process 单步 step
        '''
        coeff = (self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t])
        epsilon_theta = self.NN_predict(condition, x_t, t)
        mean = (1 / (1 - self.betas[t]).sqrt()) * (x_t - (coeff * epsilon_theta))
        sigma_t = self.betas[t].sqrt()

        # reparameterazation
        z = torch.randn_like(x_t).to(condition[0].device)
        sample = mean + z * sigma_t
        return sample

    def p_sample_loop(self, shape, condition):
        from tqdm import tqdm
        cur_x = torch.randn(shape).to(condition[0].device)
        cur_x_cpu = deepcopy(cur_x).to(torch.device("cpu"))
        x_seq = [cur_x_cpu]
        for t in tqdm(list(reversed(range(self.config["T_flow"])))):
            t = torch.LongTensor([t]).to(condition[0].device)
            cur_x = self.p_sample(cur_x, t, condition)
            cur_x_cpu = deepcopy(cur_x).to(torch.device("cpu"))
            x_seq.append(cur_x_cpu)
        return x_seq

    def DDIM_sample(self, x_t, t, t_pre, condition):
        epsilon_theta = self.NN_predict(condition, x_t, t).to(self.sqrt_alphas_cumprod[t].device)
        x_t = x_t.to(self.sqrt_alphas_cumprod[t].device)
        x0_hat = (x_t - epsilon_theta * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]

        coef_x0_hat = self.sqrt_alphas_cumprod[t_pre]
        sigma_t = self.config["DDIM_eta"] * ((1 - self.alphas_cumprod[t_pre]) / (1 - self.alphas_cumprod[t]) * (1 - self.alphas_cumprod[t] / self.alphas_cumprod[t_pre])).sqrt()
        coef_epsilon_theta = (1 - self.alphas_cumprod[t_pre] - sigma_t **2).sqrt()

        x_t_minus_1 = coef_x0_hat * x0_hat + coef_epsilon_theta * epsilon_theta + torch.randn_like(x_t, device=epsilon_theta.device) * sigma_t

        return x_t_minus_1

    def DDIM_sample_loop(self, shape, condition):
        skip = self.config["T_flow"] // self.config["DDIM_T_sample"]
        sample_Ts = list(np.array(list(range(0, self.config["T_flow"], skip))) + 1)

        sample_Ts_pre = [0] + sample_Ts[:-1]

        cur_x = torch.randn(shape).to(condition[0].device)
        topo = condition[3]
        zero_flow_idx = topo == 0
        cur_x[zero_flow_idx] = 0 # This is consistent with the diffusion process.
        x_seq = [cur_x]
        from tqdm import tqdm
        for t, t_pre in tqdm(list(zip(reversed(sample_Ts), reversed(sample_Ts_pre)))):
            t = torch.LongTensor([t]).to(condition[0].device)
            t_pre = torch.LongTensor([t_pre]).to(condition[0].device)
            cur_x = self.DDIM_sample(cur_x, t, t_pre, condition)
            cur_x_cpu = cur_x.to(torch.device("cpu"))
            x_seq.append(cur_x_cpu)
        return x_seq