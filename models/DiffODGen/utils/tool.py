import math
import json
import os
import sys
import random
from pprint import pprint

from scipy.stats import boxcox
from scipy.special import inv_boxcox

import numpy as np
import torch
import dgl
import networkx as nx




def collate_fn(samples):
    # samples is a list of pairs (graph, *)
    geoids, graphs, dises, ods = map(list, zip(*samples))

    # construct batched graph
    batched_graph = dgl.batch(graphs)
    # batched_graph = dgl.add_self_loop(batched_graph)
    # construct batched distance matrix
    dim = sum([x.shape[0] for x in dises])
    batched_dis = torch.full((dim, dim), 2, dtype=torch.float32)
    l, r = 0, 0
    for dis in dises:
        l = r
        r = r + dis.shape[0]
        batched_dis[l:r, l:r] = dis

    return geoids, batched_graph, batched_dis, ods

def od_to_topo(od):
    od[od<1] = 0
    if type(od) != type(np.array([1,1])):
        od = od.cpu().numpy()
    
    idx = od.nonzero()
    topo = np.zeros(od.shape)
    topo[idx] = 1
    return torch.FloatTensor(topo)

def trace_to_zero(od):
    for i in range(od.shape[0]):
        od[i,i] = 0
    return od

def map_location_dict(state_dict, config):
    if "device" not in config.keys():
        devices = []
        for k in state_dict.keys():
            d_id = state_dict[k].device.index
            if d_id in devices:
                continue
            devices.append(d_id)
        # if len(devices) != len(config["devices"]):
        #     raise("Device Error: devices number of state dict is not equal to the number in config!")
        map_dict = {}
        for i in range(len(devices)):
            old = "cuda:" + str(int(devices[i]))
            new = "cuda:" + str(int(config["devices"][i].index))
            map_dict[old] = new
    elif "device" in config.keys():
        devices = []
        for k in state_dict.keys():
            d_id = state_dict[k].device.index
            if d_id in devices:
                continue
            devices.append(d_id)
        map_dict = {}
        for i in range(len(devices)):
            old = "cuda:" + str(int(devices[i]))
            new = "cuda:" + str(int(config["device"].index))
            map_dict[old] = new
    return map_dict

def compute_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E，在我这里应该只有E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    # 我的这里应该是没有bs这个维度
    X_t = X_t.flatten(start_dim=0, end_dim=-2).to(torch.float32)            # N x dt
    
    Qt_T = Qt.transpose(-1, -2)                 # dt, d_t-1
    left_term = X_t @ Qt_T                      # N, d_t-1
    left_term = left_term.unsqueeze(dim=1)      # N, 1, d_t-1 这里为什么要增加这个维度呢？

    right_term = Qsb.unsqueeze(0)               # 1, d0, d_t-1
    numerator = left_term * right_term          # N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # dt, N
    
    prod = Qtb @ X_t_transposed                 # d0, N = d0, dt @ dt, N
    prod = prod.transpose(-1, -2)               # N, d0
    denominator = prod.unsqueeze(-1)            # N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator               # N, d0, d_t-1 / N, d0, 1
    return out


class DiscreteUniformTransition:
    '''
    用于topo生成的 transition matrix, e.g., Q的构成，来源于DiGress
    '''
    def __init__(self, e_classes: int):
        self.E_classes = e_classes

        self.u_e = torch.ones(self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

    def get_Qt(self, beta_t):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (float)                         noise level between 0 and 1
        returns: qe (de, de).
        """
        q_e = beta_t * self.u_e.to(beta_t.device) + (1 - beta_t) * torch.eye(self.E_classes).to(beta_t.device)

        return q_e

    def get_Qt_bar(self, alpha_bar_t):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (float)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qe (de, de).
        """
        q_e = alpha_bar_t * torch.eye(self.E_classes).to(alpha_bar_t.device) + (1 - alpha_bar_t) * self.u_e.to(alpha_bar_t.device)

        return q_e


class MarginalUniformTransition:
    '''
    用于topo生成的 transition matrix, e.g., Q的构成，来源于DiGress
    '''
    def __init__(self, e_marginals):
        self.E_classes = len(e_marginals)
        self.e_marginals = e_marginals

        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1)

    def get_Qt(self, beta_t):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (float)                         noise level between 0 and 1
        returns: qe (bs, de, de). """
        q_e = beta_t * self.u_e.to(beta_t.device) + (1 - beta_t) * torch.eye(self.E_classes).to(beta_t.device)

        return q_e

    def get_Qt_bar(self, alpha_bar_t):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (float)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qe (bs, de, de).
        """
        q_e = alpha_bar_t * torch.eye(self.E_classes).to(alpha_bar_t.device) + (1 - alpha_bar_t) * self.u_e.to(alpha_bar_t.device)

        return q_e


def node_feat_from_adj(adj):
    adj = adj.float()
    in_degree = adj.sum(0).unsqueeze(-1) / adj.sum(0).mean()
    out_degree = adj.sum(1).unsqueeze(-1) / adj.sum(1).mean()
    in_max = adj.max(0)[0].unsqueeze(-1)
    in_min = adj.min(0)[0].unsqueeze(-1)
    out_max = adj.max(1)[0].unsqueeze(-1)
    out_min = adj.min(1)[0].unsqueeze(-1)
    in_mean = adj.mean(0).unsqueeze(-1)
    out_mean = adj.mean(1).unsqueeze(-1)

    node_feats = torch.concat((in_degree, out_degree, in_max, in_min, out_max, out_min, in_mean, out_mean), dim=-1)
    return node_feats

def timestep_embedding(config, timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def rescale_od(od_normed, minmax):
    max_ = minmax["max"]
    min_ = minmax["min"]
    origin_od = (od_normed + 1) * (max_ - min_) / 2 + min_
    return origin_od

def scale_od(od):
    max_ = od.max()
    min_ = od.min()
    od_normed = (od - min_) *2 / (max_ - min_) -1
    return od_normed, {"max": max_, "min": min_}

def ToThTensor(v):
    return torch.FloatTensor(v)

def gpu_info(mem_need = 10000):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')

    mem_idx = [2, 6, 10, 14, 18, 22, 26, 30]
    mem_bus = [x for x in range(8)]
    mem_list = []
    for idx, info in enumerate(gpu_status):
        if idx in mem_idx:
            mem_list.append(11019 - int(info.split('/')[0].split('M')[0].strip()))
    idx = np.array(mem_bus).reshape([-1, 1])
    mem = np.array(mem_list).reshape([-1, 1])
    id_mem = np.concatenate((idx, mem), axis=1)
    GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]

    if len(GPU_available) != 0:
        return GPU_available
    else:
        return None

def narrow_setup(interval = 0.5, mem_need = 10000):
    GPU_available = gpu_info()
    i = 0
    while GPU_available is None:  # set waiting condition
        GPU_available = gpu_info(mem_need)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        # time.sleep(interval)
        i += 1
    GPU_selected = random.choice(GPU_available)
    return GPU_selected

def get_conifg(path):
    config = json.load(open(path, "r"))
    print("\n****** experiment name:", config["exp_name"], " ******")

    # check GPU available
    if config["check_device"] == 1:
        GPU_no = narrow_setup(interval = 1, mem_need = 8100)
        config["device"] = torch.device(int(GPU_no))
        print("\n****** Using No.", int(GPU_no), "GPU ******")

    print("\n", "****** exp config ******")
    pprint(config)
    print("*************************\n")
    return config

def build_DGLGraph(adj):
    edges = adj.nonzero()
    edges = list(zip(edges[0].tolist(), edges[1].tolist()))
    g_nx = nx.Graph(edges)
    g_nx.add_nodes_from([x for x in range(adj.shape[0])])
    g = dgl.from_networkx(g_nx)
    g = dgl.add_self_loop(g)
    return g

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class None_transformer():
    def __init__(self):
        pass

    def fit_transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x


class Log_transformer():
    def __init__(self):
        pass

    def fit_transform(self, x):
        return np.log1p(x)
    
    def inverse_transform(self, x):
        return np.expm1(x)


class BoxCox_transformer():
    def __init__(self):
        self.lmbda0 = None

    def fit_transform(self, x):
        shape = x.shape
        x = x.reshape([-1])
        x = x + 1
        x, lambda0 = boxcox(x, lmbda=None, alpha=None)
        self.lmbda0 = lambda0
        x = x.reshape(shape)
        return x
    
    def inverse_transform(self, x):
        shape = x.shape
        x = inv_boxcox(x, self.lmbda0) - 1
        x = x.reshape(shape)
        return x


def test():
    # betas = gen_betas_flow()
    # alphas = get_alphas_flow(betas)
    # betas_bar = get_coeff_bars(alphas)

    # betas = get_named_beta_schedule("cosine", 100)

    # print(betas, betas.shape)
    path = "/data/rongcan/code/od-diffusion/exp/config/NYC_Chi.json"
    get_conifg(path)

if __name__ == "__main__":
    test()
    
