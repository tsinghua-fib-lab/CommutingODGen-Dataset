import os
import sys
import math

from tqdm import tqdm
from random import shuffle
from multiprocessing import Pool

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import torch
import dgl

from utils.metrics import *





def mean_the_denoising_process(seqs):
    seqs_npy = []
    for seq in seqs:
        seq_npy = []
        for net_t in seq:
            _, e_t = net_t
            seq_npy.append(e_t.numpy())
        seq_npy = np.stack(seq_npy)
        seqs_npy.append(seq_npy)
    seqs_npy = np.stack(seqs_npy)
    
    seqs_npy = np.mean(seqs_npy, axis=0)
    return seqs_npy


def plot_od_arc_chart(od, geometries):
    from geopandas import GeoDataFrame
    from shapely.geometry import LineString
    import contextily as cx
    import matplotlib
    import matplotlib.pyplot as plt

    font = {'size': 12}
    matplotlib.rc('font', **font)

    low, high = 15, 80
    
    OD = od
    point = geometries.centroid
    line = []
    for i in tqdm(range(OD.shape[0])):
        for j in range(OD.shape[1]):
            if i != j and OD[i][j] > 0:
                line.append( [point[i], point[j], OD[i][j]] )
    points_df = GeoDataFrame(line, columns = ['geometry_o', 'geometry_d', 'flow'])
    points_df['line'] = points_df.apply(lambda x: LineString([x['geometry_o'], x['geometry_d']]), axis=1)

    f, ax = plt.subplots(1,figsize=(6,8),dpi=200)

    target_gdf = GeoDataFrame(points_df[points_df['flow']<=low], geometry=points_df['line'])
    target_gdf.crs = "EPSG:4326"
    target_gdf.plot(ax = ax, column = 'flow',linewidth=0.05,color='#0308F8',alpha = 0.4)

    target_gdf = GeoDataFrame(points_df[(points_df['flow']>low) & (points_df['flow']<=high)], geometry=points_df['line'])
    target_gdf.plot(ax=ax, column = 'flow',linewidth=0.1,color='#FD0B1B',alpha = 0.6)

    target_gdf = GeoDataFrame(points_df[points_df['flow']>high], geometry=points_df['line'])
    target_gdf.crs = "EPSG:4326"
    target_gdf.plot(ax=ax, column = 'flow', linewidth=0.15, color='yellow',alpha=0.8)


    cx.add_basemap(ax, crs=target_gdf.crs,source=cx.providers.CartoDB.Positron)

    # plt.minorticks_off()
    plt.xticks([])
    plt.yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return f

def feat_to_pairs(nfeat, dis):
    ofeat = nfeat[:, None, :].repeat(1, nfeat.shape[0], 1)
    dfeat = nfeat[None, :, :].repeat(nfeat.shape[0], 1, 1)
    dis = dis[:, :, None]
    feat = torch.cat([ofeat, dfeat, dis], dim=2)
    return feat


def net_to_pairs(nfeat, dis, od):
    # feat
    ofeat = nfeat[:, np.newaxis, :].repeat(nfeat.shape[0], axis=1)
    dfeat = nfeat[np.newaxis, :, :].repeat(nfeat.shape[0], axis=0)
    dis = dis[:, :, np.newaxis]
    feat = np.concatenate([ofeat, dfeat, dis], axis=2)
    feat = feat.reshape([-1, feat.shape[-1]])
    # od
    od = od.reshape([-1, 1])
    return feat, od


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


def SegLog_transformer():
    def __init__(self):
        self.segs = [5, 10, 20, 50, 100, 200, 500, 1000]

    def fit_transform(self, x):
        x = x + 1
        l, r = 0, 0
        for i in range(len(self.segs)):
            l = r
            r = self.segs[i]
            if i == 0:
                continue
            x[(l<=x) and (x<r)] = log(i+1, (x[(l<=x) and (x<r)]))
        return np.log1p(x)
    
    def inverse_transform(self, x):
        return np.expm1(x)


def MinMaxer(data):
    return MinMaxScaler(feature_range=(-1, 1)).fit(data)


def log(a, b):
    return np.log(a) / np.log(b)


def reshape_matrix(matrix, k=100):
    n = matrix.shape[0]
    
    if k > n:
        padding = np.zeros((n, k-n))
        return np.hstack([matrix, padding])
    if k < n:
        return matrix[:, :k]
    return matrix


def compute_Sim_LaPE_of_one_city(k, dis, kernal=10000):
    A = np.exp(-(dis **2 / (2 * np.power(kernal, 2))))

    out = np.sum(A, axis=1)
    D = np.diag(out)
    L = D - A

    _, eigenvectors = np.linalg.eig(L) # eigenvalues, eigenvectors

    LaPE = reshape_matrix(eigenvectors.T, k=k)
    return LaPE


def compute_LaPE_of_one_city(k, dis):
    A = dis

    out = np.sum(A, axis=1)
    D = np.diag(out)
    L = D - A

    _, eigenvectors = np.linalg.eig(L) # eigenvalues, eigenvectors
    LaPE = reshape_matrix(eigenvectors.T, k=k)
    return LaPE


def get_LaPE_from_dis(config, dis, batchlization):
    LaPEs = []
    l, r = 0, 0
    for _, i in enumerate(recover_od_shapes(batchlization)):
        l = r
        r = r + i
        dis_tmp = dis[l:r, l:r].cpu().numpy()
        LaPE = compute_LaPE_of_one_city(config["LaPE_dim"], dis_tmp)
        sim_LaPE = compute_Sim_LaPE_of_one_city(config["LaPE_dim"], dis_tmp)
        LaPE = torch.cat([torch.FloatTensor(LaPE), torch.FloatTensor(sim_LaPE)], dim=-1)
        LaPEs.append(LaPE)
    LaPEs = torch.cat(LaPEs).to(config["device"])
    return LaPEs


def process_subtask(k, dis_tmp):
    LaPE = compute_LaPE_of_one_city(k, dis_tmp)
    sim_LaPE = compute_Sim_LaPE_of_one_city(k, dis_tmp)
    LaPE = np.concatenate([LaPE, sim_LaPE], axis=-1)
    return LaPE


def get_LaPE_from_dis_parallel(config, dis, batchlization): # speed up with parallel
    tasks = []
    l, r = 0, 0
    for _, i in enumerate(recover_od_shapes(batchlization)):
        l = r
        r = r + i
        dis_tmp = dis[l:r, l:r].cpu().numpy()
        tasks.append((config["LaPE_dim"], dis_tmp))
    with Pool(processes=len(tasks)) as pool:
        LaPEs = pool.starmap(process_subtask, tasks)
    LaPEs = [torch.FloatTensor(LaPE) for LaPE in LaPEs]
    # LaPEs = []
    # for task in tasks:
    #     k, dis_tmp = task
    #     LaPE = process_subtask(k, dis_tmp)
    #     LaPEs.append(LaPE)
    # LaPEs = [torch.FloatTensor(LaPE) for LaPE in LaPEs]
    return torch.cat(LaPEs).to(config["device"])


def recover_od_shapes(batchlization):
    shapes = []
    i = 0
    while i < batchlization.shape[0]:
        if batchlization[i, i] == 1:
            count = 1
            while i + count < batchlization.shape[0] and batchlization[i, i + count] == 1:
                count += 1
            shapes.append(count)
            i += count
        else:
            i += 1
    return shapes


def redirect_output_to_file(exp_name):
    output_file = f"exp/stdout/{exp_name}.txt"
    sys.stdout.flush()
    sys.stderr.flush()
    with open(output_file, 'w') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())


def split_data_intoTVT(data, config):
    if (config["train_set"] < 1) and (config["valid_set"] < 1) and (config["test_set"] < 1):
        train = data[:int(len(data)*config["train_set"])]
        valid = data[int(len(data)*config["train_set"]): int(len(data)*config["train_set"])+int(len(data)*config["valid_set"])]
        test = data[int(len(data)*config["train_set"])+int(len(data)*config["valid_set"]):]
    else:
        train = data[:config["train_set"]]
        valid = data[-(config["test_set"]+config["valid_set"]):-config["test_set"]]
        test = data[-config["test_set"]:]
        if len(train) + len(valid) + len(test) > len(data):
            raise Exception('The samples in train, valid and test are more than raw data!')
    return train, valid, test


def permut_nodes_order(g, dis, od):
    permute_idx = [x for x in range(dis.shape[0])]
    shuffle(permute_idx)

    g = dgl.reorder_graph(g, node_permute_algo='custom', permute_config={'nodes_perm': permute_idx}, store_ids=True)
    dis = dis[permute_idx, :][:, permute_idx]
    od = od[permute_idx, :][:, permute_idx]
    
    return g, dis, od

def permut_nodes_order_back(inputs, masks, dis, block_mask):
    input_n, input_e = inputs
    mask_n, mask_e = masks

    permute_idx = [x for x in range(input_n.shape[0])]
    shuffle(permute_idx)
    input_n, input_e = input_n[permute_idx, :, :], input_e[permute_idx, :, :][:, permute_idx, :]
    mask_n, mask_e = mask_n[permute_idx, :], mask_e[permute_idx, :][:, permute_idx]
    dis = dis[permute_idx, :][:, permute_idx]
    block_mask = block_mask[permute_idx, :][:, permute_idx]
    inputs = (input_n, input_e)
    masks = (mask_n, mask_e)

    return inputs, masks, dis, block_mask

def trace_to_zero(od):
    for i in range(od.shape[0]):
        od[i,i] = 0
    return od


def timestep_embedding(config, timesteps, dim, max_period=10000, batchlization=None):
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
    if len(timesteps) == 1:
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    else:
        embedding_for_n = torch.zeros([batchlization.shape[0], dim]).to(device=timesteps.device)
        embedding_for_e = torch.zeros([batchlization.shape[0], batchlization.shape[1], dim]).to(device=timesteps.device)
        l, r = 0, 0
        for idx, i in enumerate(recover_od_shapes(batchlization)):
            l = r
            r = r + i
            args = timesteps[:,None][idx][:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            embedding_tmp_for_n = embedding.expand(i, -1)
            embedding_tmp_for_e = embedding[None, :, :].expand(i, i, -1)
            embedding_for_n[l:r] = embedding_tmp_for_n
            embedding_for_e[l:r, l:r] = embedding_tmp_for_e
        return embedding_for_n, embedding_for_e



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


def generate_masks(config, n, e):
    # 0 means there is no information, which need imputation or generation

    #### mask_n
    random_number = np.random.rand()
    if random_number < config["p_featMissing"]:
        # some feature missing exists
        seed = np.random.rand()
        mask_n = np.random.rand(*list(n.shape))
        mask_n[mask_n > seed] = 1
        mask_n[mask_n <= seed] = 0
    else:
        # no feature missing
        mask_n = np.ones(list(n.shape))
    mask_n = torch.FloatTensor(mask_n)

    #### mask_e
    random_number = np.random.rand()
    if random_number < config["p_generation"]:
        # generation
        mask_e = np.zeros(list(e.shape))
    else:
        # completion
        seed = np.random.rand()
        mask_e = np.random.rand(*list(e.shape))
        mask_e[mask_e > seed] = 1
        mask_e[mask_e <= seed] = 0
    mask_e = torch.FloatTensor(mask_e)

    return mask_n, mask_e


def generate_eval_mask(config, n, e):
    # 0 means there is no information, which need imputation or generation

    #### mask_n
    random_number = np.random.rand()
    if random_number < config["p_featMissing"]:
        # some feature missing exists
        seed = 0.5
        mask_n = np.random.rand(*list(n.shape))
        mask_n[mask_n > seed] = 1
        mask_n[mask_n <= seed] = 0
        miss_flag = True
    else:
        # no feature missing
        mask_n = np.ones(list(n.shape))
        miss_flag = False
    mask_n = torch.FloatTensor(mask_n)

    #### mask_e
    random_number = np.random.rand()
    if random_number < config["p_generation"]:
        # generation
        mask_e = np.zeros(list(e.shape))
        gen_flag = True
    else:
        # completion
        seed = 0.5
        mask_e = np.random.rand(*list(e.shape))
        mask_e[mask_e > seed] = 1
        mask_e[mask_e <= seed] = 0
        gen_flag = False
    mask_e = torch.FloatTensor(mask_e)

    flags = (miss_flag, gen_flag)
    return mask_n, mask_e, flags

def generate_Gen_mask(n, e):
    mask_n = np.ones(list(n.shape))
    mask_e = np.zeros(list(e.shape))

    mask_n = torch.FloatTensor(mask_n)
    mask_e = torch.FloatTensor(mask_e)
    return mask_n, mask_e

def generate_Comp_mask(n, e):
    mask_n = np.ones(list(n.shape))
    
    seed = 0.5
    mask_e = np.random.rand(*list(e.shape))
    mask_e[mask_e > seed] = 1
    mask_e[mask_e <= seed] = 0

    mask_n = torch.FloatTensor(mask_n)
    mask_e = torch.FloatTensor(mask_e)
    return mask_n, mask_e

def generate_MissGen_mask(n, e):
    seed = 0.5
    mask_n = np.random.rand(*list(n.shape))
    mask_n[mask_n > seed] = 1
    mask_n[mask_n <= seed] = 0

    mask_e = np.zeros(list(e.shape))

    mask_n = torch.FloatTensor(mask_n)
    mask_e = torch.FloatTensor(mask_e)
    return mask_n, mask_e

def generate_MissComp_mask(n, e):
    seed = 0.5
    mask_n = np.random.rand(*list(n.shape))
    mask_n[mask_n > seed] = 1
    mask_n[mask_n <= seed] = 0

    seed = 0.5
    mask_e = np.random.rand(*list(e.shape))
    mask_e[mask_e > seed] = 1
    mask_e[mask_e <= seed] = 0

    mask_n = torch.FloatTensor(mask_n)
    mask_e = torch.FloatTensor(mask_e)
    return mask_n, mask_e

class validEvaluator():
    def __init__(self, config):
        self.config = config

        self.feat_metrics = []
        self.od_metrics = {"gen": [],
                           "comp": [],
                           "miss_gen": [],
                           "miss_comp": []}

    def batch_tag(self, flags):
        miss_flag, gen_flag = flags
        if miss_flag and gen_flag:
            self.tag = "miss_gen"
        elif (not miss_flag) and gen_flag:
            self.tag = "gen"
        elif (not miss_flag) and (not gen_flag):
            self.tag = "comp"
        elif miss_flag and (not gen_flag):
            self.tag = "miss_comp"
        elif self.tag not in ["miss_gen", "gen", "comp", "miss_comp"]:
            raise Exception("Wrong tag!")

    def evaluate_batch(self, pred, gt, masks, flags, scalers):
        self.batch_tag(flags)
        n_hat, e_hat = pred
        n, ods = gt
        mask_n, mask_e = masks

        l, r = 0, 0
        for _, od in enumerate(ods):
            l = r
            r = r + od.shape[0]
            # extract
            nfeat = n[l:r, :]
            nfeat_hat = n_hat[l:r, :]
            od_hat = e_hat[l:r, l:r]
            tmp_mask_n = mask_n[l:r, :]
            tmp_mask_e = mask_e[l:r, l:r]

            # feat
            if self.tag not in ["comp", "gen"]:
                nfeat = scalers["feat"].inverse_transform(nfeat.cpu().numpy())
                nfeat[nfeat < 0] = 0
                nfeat_hat = scalers["feat"].inverse_transform(nfeat_hat)
                nfeat_hat[nfeat_hat < 0] = 0
                nfeat_hat[tmp_mask_n != 0] = nfeat[tmp_mask_n != 0]
                feat_metrics = cal_feat_metrics(nfeat_hat, nfeat)
                self.feat_metrics.append(feat_metrics)
            
            # od
            if np.sum(tmp_mask_e == 0) == 0:
                continue
            od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
            od = scalers["od_normer"].inverse_transform(od)
            od[od < 0] = 0
            od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
            od_hat = scalers["od_normer"].inverse_transform(od_hat)
            od_hat[od_hat < 0] = 0
            for i in range(od_hat.shape[0]): # consistant with groundtruth
                od_hat[i,i] = 0
            od_hat = np.floor(od_hat)
            od_metrics = cal_od_metrics(od_hat, od)
            if self.tag in ["comp", "miss_comp"]:
                od = od[tmp_mask_e == 0]
                od_hat = od_hat[tmp_mask_e == 0]
                od_metrics = cal_NonMatrix_metrics(od_hat, od)
            self.od_metrics[self.tag].append(od_metrics)

    def summary_all_metrics(self):
        feat = average_listed_metrics(self.feat_metrics)
        gen = average_listed_metrics(self.od_metrics["gen"])
        comp = average_listed_metrics(self.od_metrics["comp"])
        miss_gen = average_listed_metrics(self.od_metrics["miss_gen"])
        miss_comp = average_listed_metrics(self.od_metrics["miss_comp"])

        seg_feat = citywise_segmented_metrics(self.feat_metrics)
        seg_gen = citywise_segmented_metrics(self.od_metrics["gen"])
        seg_comp = citywise_segmented_metrics(self.od_metrics["comp"])
        seg_miss_gen = citywise_segmented_metrics(self.od_metrics["miss_gen"])
        seg_miss_comp = citywise_segmented_metrics(self.od_metrics["miss_comp"])

        all_metrics = {
            "feat" : seg_feat,
            "gen" : seg_gen,
            "comp" : seg_comp,
            "miss_gen" : seg_miss_gen,
            "miss_comp" : seg_miss_comp
        }

        all_metrics["feat"]["all"] = feat
        all_metrics["gen"]["all"] = gen
        all_metrics["comp"]["all"] = comp
        all_metrics["miss_gen"]["all"] = miss_gen
        all_metrics["miss_comp"]["all"] = miss_comp

        return all_metrics

