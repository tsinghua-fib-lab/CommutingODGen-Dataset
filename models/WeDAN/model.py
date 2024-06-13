import math
from copy import deepcopy
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.tool import *



class PairWise_PreModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_in = nn.Linear(config["n_indim"] *2 +1, 64)

        self.linears = nn.ModuleList(
            [nn.Linear(64, 64) for i in range(5)]
        )
        self.linear_out = nn.Linear(64, 1)

    def forward(self, input):
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x))
        x = x + input
        x = torch.tanh(self.linear_out(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["n_indim"]
        self.n_head = config["num_head_cross"]
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

        weighted_V = weighted_V + x
        
        return weighted_V


class NodeEdgeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["hiddim"]
        edim = config["hiddim"]
        self.n_head = config["num_head"]
        self.df = int(xdim / self.n_head)

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

    def forward(self, x, e, batchlization):
        # Map X to keys and queries
        Q = self.q(x)
        K = self.k(x)
        
        # Reshape to (n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        
        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)

        # Compute unnormalized attentions.
        Y = Q * K                                      # (n, n, n_head, df)
        Y = Y / math.sqrt(Y.size(-1))                  # (n, n, n_head, df)
        
        E1 = self.e_mul(e)                             # (n, n, dx)
        E1 = E1.reshape((e.size(0), e.size(1), self.n_head, self.df))
        
        E2 = self.e_add(e)
        E2 = E2.reshape((e.size(0), e.size(1), self.n_head, self.df))
        
        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                          # (n, n, n_head, df)

        # remove the effect of the node pairs from batchlization of graphs
        Y[batchlization == 0] = -9e9
        
        # Compute attentions. attn is still (n, n, n_head, df)
        attn = F.softmax(Y, dim=1)
        
        # Map X to values
        V = self.v(x)                        # n, dx
        V = V.reshape((V.size(0), self.n_head, self.df))
        V = V.unsqueeze(0)                   # (1, n, n_head, df)
        
        # Compute weighted values
        weighted_V = attn * V                # (n, n, n_head, df)
        weighted_V = weighted_V.sum(dim=1)   # (n, n_head, df)
        
        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=1)            # (n, dx)

        # Output X
        newX = self.x_out(weighted_V)

        # Output E
        Y[batchlization == 0] = -2
        newE = Y.flatten(start_dim=2)
        newE = self.e_out(newE)
        
        return newX, newE
    

class FeatureTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        hiddim = config["hiddim"]
        num_head = config["num_head"]
        dropout = config["dropout"]

        dim_ff = config["hiddim"]

        self.xin = nn.Linear(1, hiddim)
        self.self_attn = nn.MultiheadAttention(hiddim, num_head, dropout, batch_first=True)

        self.lin1 = nn.Linear(hiddim, dim_ff)
        self.lin2 = nn.Linear(dim_ff, hiddim)
        if config["norm_type"] == "batch":
            self.norm1 = nn.BatchNorm1d(config["n_indim"]) # LayerNorm(hiddim)
            self.norm2 = nn.BatchNorm1d(config["n_indim"]) # LayerNorm(hiddim)
        elif config["norm_type"] == "layer":
            self.norm1 = nn.LayerNorm(hiddim)
            self.norm2 = nn.LayerNorm(hiddim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.xout = nn.Linear(hiddim, 1)

    def get_position_embedding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_emb = position * div_term
        pos_emb[:, 0::2] = torch.sin(pos_emb[:, 0::2])  # dim 2i
        pos_emb[:, 1::2] = torch.cos(pos_emb[:, 1::2])  # dim 2i+1
        return pos_emb

    def add_posEmb(self, x):
        pos_emb = self.get_position_embedding(x.shape[1], x.shape[2]*2)
        x = x + pos_emb.to(x.device)
        return x

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.xin(x)
        x = self.add_posEmb(x)

        new_x, _ = self.self_attn(x, x, x, need_weights=False)

        new_x = self.dropout1(new_x)
        x = self.norm1(x + new_x)

        ff_output = self.lin2(self.dropout2(F.relu(self.lin1(x), inplace=True)))
        ff_output = self.dropout3(ff_output)
        x = self.norm2(x + ff_output)

        x = self.xout(x).squeeze()
        return x

class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        xdim, edim = config["hiddim"], config["hiddim"]
        dim_ffX, dim_ffE = config["hiddim"], config["hiddim"]

        self.self_attn = NodeEdgeBlock(config)

        self.linX1 = nn.Linear(xdim, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, xdim)
        if config["norm_type"] == "batch":
            self.normX1 = nn.BatchNorm1d(xdim)
            self.normX2 = nn.BatchNorm1d(xdim)
        elif config["norm_type"] == "layer":
            self.normX1 = nn.LayerNorm(xdim)
            self.normX2 = nn.LayerNorm(xdim)
        self.dropoutX1 = nn.Dropout(config["dropout"])
        self.dropoutX2 = nn.Dropout(config["dropout"])
        self.dropoutX3 = nn.Dropout(config["dropout"])

        self.linE1 = nn.Linear(edim, dim_ffE)
        self.linE2 = nn.Linear(edim, dim_ffE)
        if config["norm_type"] == "batch":
            self.normE1 = nn.BatchNorm1d(edim)
            self.normE2 = nn.BatchNorm1d(edim)
        elif config["norm_type"] == "layer":
            self.normE1 = nn.LayerNorm(edim)
            self.normE2 = nn.LayerNorm(edim)
        self.dropoutE1 = nn.Dropout(config["dropout"])
        self.dropoutE2 = nn.Dropout(config["dropout"])
        self.dropoutE3 = nn.Dropout(config["dropout"])


    def forward(self, x, e, batchlization):
        #### self attention
        newX, newE = self.self_attn(x, e, batchlization)
        
        newX_d = self.dropoutX1(newX) # (n, h)
        x = self.normX1(x + newX_d)

        newE_d = self.dropoutE1(newE) # (n, n, h)
        if self.config["norm_type"] == "layer":
            e = self.normE1(e + newE_d)
        elif self.config["norm_type"] == "batch":
            e = e + newE_d
            e = self.normE1(e.reshape([-1, e.shape[-1]])).reshape([e.shape[0], e.shape[1], e.shape[2]])

        #### FFN
        ff_outputX = self.linX2(self.dropoutX2(F.relu(self.linX1(x), inplace=True)))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(x + ff_outputX)
        
        ff_outputE = self.linE2(self.dropoutE2(F.relu(self.linE1(e), inplace=True)))
        ff_outputE = self.dropoutE3(ff_outputE)
        if self.config["norm_type"] == "layer":
            E = self.normE2(e + ff_outputE)
        elif self.config["norm_type"] == "batch":
            E = e + ff_outputE
            E = self.normE2(E.reshape([-1, E.shape[-1]])).reshape([E.shape[0], E.shape[1], E.shape[2]])
        
        return X, E
    

class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        x_indim = config["n_indim"] * 2 # obervation, missing
        e_indim = config["e_indim"]
        x_hiddim = config["hiddim"]
        e_hiddim = config["hiddim"]
        x_outdim = config["n_outdim"]
        e_outdim = config["e_outdim"]

        self.x_outdim = x_outdim
        self.e_outdim = e_outdim

        self.xin = nn.Linear(x_indim, x_hiddim)
        self.ein = nn.Linear(e_indim, e_hiddim)

        self.gtlayers = nn.ModuleList()
        self.n_invs = nn.ModuleList()
        self.nftlayers = nn.ModuleList()
        self.crossAttns = nn.ModuleList()
        self.n_projs = nn.ModuleList()
        for _ in range(config["num_layer"]):
            self.gtlayers.append(GraphTransformerLayer(config))
            self.n_invs.append(nn.Linear(config["hiddim"], config["n_indim"]))
            self.nftlayers.append(FeatureTransformerLayer(config))
            self.crossAttns.append(CrossAttention(config))
            self.n_projs.append(nn.Linear(config["n_indim"], config["hiddim"]))
        
        self.xout = nn.Sequential(
            nn.Linear(x_hiddim, x_hiddim), nn.ReLU(inplace=True),
            nn.Linear(x_hiddim, x_outdim)
        )
        self.eout = nn.Sequential(
            nn.Linear(e_hiddim, e_hiddim), nn.ReLU(inplace=True),
            nn.Linear(e_hiddim, e_outdim)
        )

    def forward(self, x, e, c, batchlization):
        t_emb, n_c, e_c = c

        x_h, e_h = self.xin(x), self.ein(e)
        # skip_x, skip_e = [], []
        for i in range(self.config["num_layer"]):
            x_h_res, e_h_res = x_h, e_h

            # t
            if not isinstance(t_emb, tuple):
                x_h, e_h = x_h + t_emb, e_h + t_emb.unsqueeze(0)
            else:
                x_h, e_h = x_h + t_emb[0], e_h + t_emb[1]

            # transformer
            x_h, e_h = self.gtlayers[i](x_h, e_h, batchlization)
            # add c
            x_h = self.n_projs[i](self.nftlayers[i](self.crossAttns[i](self.n_invs[i](x_h), n_c)))
            e_h = e_h + e_c
            # res
            x_h = x_h + x_h_res
            e_h = e_h + e_h_res
        
        X = self.xout(x_h) + x[..., :self.config["n_outdim"]]
        E = self.eout(e_h) + e[..., :self.config["e_outdim"]]
        
        return X, E
    
class ResidualLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config["hiddim"], config["hiddim"])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x) + x
        return x

class DenoisingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # # pretrained pairwise model provides conditions
        # self.pretrain_pairwise = PairWise_PreModel(config)
        # self.pretrain_pairwise.load_state_dict(torch.load("exp/model/pretrain/DEBUG_100_1000_pretrain_"+str(334)+".pkl", map_location=config["device"]))
        # proj the dim into hiddim
        self.N_c_proj = nn.Linear(config["n_indim"]+2*config["LaPE_dim"], config["n_indim"])
        self.E_c_proj = nn.Linear(2, config["hiddim"])

        self.graph_transformer = GraphTransformer(config)

    def forward(self, inputs, masks, dis, t, batchlization):
        # net
        n, e = inputs
        n = n.flatten(start_dim=1)

        # time embedding
        if len(t) == 1:
            t_emb = timestep_embedding(self.config, t, dim=self.config["hiddim"], max_period=self.config["T"], batchlization=batchlization)
        else:
            t_emb_for_n, t_emb_for_e = timestep_embedding(self.config, t, dim=self.config["hiddim"], max_period=self.config["T"], batchlization=batchlization)
            t_emb = (t_emb_for_n, t_emb_for_e)
        
        # LaPE
        if self.config["LaPE_dim"] != 0:
            LaPE = get_LaPE_from_dis(self.config, dis, batchlization)
            # LaPE = get_LaPE_from_dis_parallel(self.config, dis, batchlization)

        # # pretrained pairwise condition
        # nfeat = feat_to_pairs(n.reshape([n.shape[0], -1, 2]).sum(-1), dis)
        # with torch.no_grad():
        #     pretrain_c = self.pretrain_pairwise(nfeat)

        n_mask, e_mask = masks
        e_mask = e_mask.unsqueeze(-1)
        dis_emb = dis.unsqueeze(-1)
        
        # condition
        if self.config["LaPE_dim"] != 0:
            n_c = self.N_c_proj(torch.cat([n_mask, LaPE], dim=-1))
        else:
            n_c = n_mask

        e_c = self.E_c_proj(torch.cat([e_mask, dis_emb], dim=-1)) # , pretrain_c
        e_c[batchlization == 0] = -2 # debug batchlization
        
        c = (t_emb, n_c, e_c) # t n and e, before transformer; n_mask as c for n; e_mask and dis as c for e

        # Graph Transformer
        pred_n, pred_e = self.graph_transformer(n, e, c, batchlization)
        pred_e = pred_e.squeeze()
        
        return pred_n, pred_e

class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config= config

        self.model = DenoisingNet(config)

        betas = get_named_beta_schedule(schedule_name=config["beta_scheduler"], num_diffusion_timesteps=config["T"])
        betas = torch.FloatTensor(betas)

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

    def q_sample(self, net, t, batchlization):
        n, e = net

        noise_n = torch.randn_like(n).to(n.device)
        noise_e = torch.randn_like(e).to(e.device)

        if len(t) == 1:
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

            noisy_n = sqrt_alphas_cumprod_t * n + sqrt_one_minus_alphas_cumprod_t * noise_n
            noisy_e = sqrt_alphas_cumprod_t * e + sqrt_one_minus_alphas_cumprod_t * noise_e
        else:
            noisy_n = torch.zeros_like(noise_n).to(n.device)
            noisy_e = torch.zeros_like(noise_e).to(n.device)
            l, r = 0, 0
            for idx, i in enumerate(recover_od_shapes(batchlization)):
                sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t[idx]]
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t[idx]]

                l = r
                r = r + i
                noisy_n[l:r]      = sqrt_alphas_cumprod_t * n[l:r]      + sqrt_one_minus_alphas_cumprod_t * noise_n[l:r]
                noisy_e[l:r, l:r] = sqrt_alphas_cumprod_t * e[l:r, l:r] + sqrt_one_minus_alphas_cumprod_t * noise_e[l:r, l:r]

        return (noisy_n, noisy_e), (noise_n, noise_e)

    def loss(self, net, masks, t, dis, batchlization):
        n, e = net
        (noisy_n, noisy_e), (noise_n, noise_e) = self.q_sample(net, t, batchlization)

        # inputs
        mask_n, mask_e = masks
        input_n = torch.cat([(n * mask_n                ).unsqueeze(-1), (noisy_n * (1 - mask_n)                ).unsqueeze(-1)], dim=-1)
        input_e = torch.cat([(e * mask_e * batchlization).unsqueeze(-1), (noisy_e * (1 - mask_e) * batchlization).unsqueeze(-1)], dim=-1)
        inputs = (input_n, input_e)

        # model
        noise_n_hat, noise_e_hat = self.model(inputs, masks, dis, t, batchlization)

        # loss
        eloss_idx = (mask_e == 0) & (batchlization == 1)
        noise_hat = torch.cat([(noise_n_hat * (1 - mask_n))[mask_n==0].reshape([-1]), (noise_e_hat * (1 - mask_e) * batchlization)[eloss_idx].reshape([-1])])
        noise     = torch.cat([(noise_n     * (1 - mask_n))[mask_n==0].reshape([-1]), (noise_e     * (1 - mask_e) * batchlization)[eloss_idx].reshape([-1])])
        
        if self.config["loss"][:13] == "flow_weighted":
            loss = self.loss_fn(noise_hat, noise, e[eloss_idx])
        else:
            loss = self.loss_fn(noise_hat, noise)
        return loss

    @property
    def loss_fn(self):
        if self.config["loss"] == "mse":
            return F.mse_loss
        elif self.config["loss"] == "l1":
            return F.l1_loss
        elif self.config["loss"] == "smooth_l1":
            return F.smooth_l1_loss
        elif self.config["loss"] == "flow_weighted_mse":
            return self.flow_weighted_mse_loss
        elif self.config["loss"] == "flow_weighted_mae":
            return self.flow_weighted_mae_loss
        else:
            raise NotImplementedError
        

    def flow_weighted_mse_loss(self, x, x_hat, flow, beta=0.1):
        losses = F.mse_loss(x, x_hat, reduction='none')
        weights = 1 / (flow - (-1) + beta)
        # weights = weights / weights.sum()
        loss = (losses * weights).mean()
        return loss
    
    def flow_weighted_mae_loss(self, x, x_hat, flow, beta=0.1):
        losses = F.l1_loss(x, x_hat, reduction='none')
        weights = 1 / (flow - (-1) + beta)
        # weights = weights / weights.sum()
        loss = (losses * weights).mean()
        return loss


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
        cur_x = torch.randn(shape).to(condition[0].device)
        cur_x_cpu = deepcopy(cur_x).to(torch.device("cpu"))
        x_seq = [cur_x_cpu]
        for t in tqdm(list(reversed(range(self.config["T_flow"])))):
            t = torch.LongTensor([t]).to(condition[0].device)
            cur_x = self.p_sample(cur_x, t, condition)
            cur_x_cpu = deepcopy(cur_x).to(torch.device("cpu"))
            x_seq.append(cur_x_cpu)
        return x_seq

    def DDIM_sample(self, cur_net, t, t_pre, c):
        # cur net
        cur_n, cur_e = cur_net
        # condition content
        real_net, masks, dis, batchlization = c # block because of batch lization
        real_n, real_e = real_net
        
        # inputs
        mask_n, mask_e = masks
        input_n = torch.cat([(real_n * mask_n                ).unsqueeze(-1), (cur_n * (1 - mask_n)                ).unsqueeze(-1)], dim=-1)
        input_e = torch.cat([(real_e * mask_e * batchlization).unsqueeze(-1), (cur_e * (1 - mask_e) * batchlization).unsqueeze(-1)], dim=-1)
        inputs = (input_n, input_e)

        # predict noise net
        noise_n_hat, noise_e_hat = self.model(inputs, masks, dis, t, batchlization)
        noise_e_hat = noise_e_hat * batchlization

        # noise net, cur_net -> net_0_hat
        n0_hat = (cur_n - noise_n_hat * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]
        e0_hat = (cur_e - noise_e_hat * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]

        # some coefficient
        coef_net0_hat = self.sqrt_alphas_cumprod[t_pre]
        sigma_t = self.config["DDIM_eta"] * ((1 - self.alphas_cumprod[t_pre]) / (1 - self.alphas_cumprod[t]) * (1 - self.alphas_cumprod[t] / self.alphas_cumprod[t_pre])).sqrt()
        coef_noise_hat = (1 - self.alphas_cumprod[t_pre] - sigma_t **2).sqrt()

        # reparameterazation
        n_t_minus_1 = coef_net0_hat * n0_hat + coef_noise_hat * noise_n_hat + torch.randn_like(cur_n, device=torch.device("cuda")) * sigma_t
        e_t_minus_1 = coef_net0_hat * e0_hat + coef_noise_hat * noise_e_hat + torch.randn_like(cur_e, device=torch.device("cuda")) * sigma_t * batchlization
        net_t_minus_1 = (n_t_minus_1, e_t_minus_1)

        return net_t_minus_1

    def DDIM_sample_loop(self, shape_n, shape_e, c):
        # block matrix mask 
        _, _, _, batchlization = c # block because of batch lization
        # DDIM setting
        skip = self.config["T"] // self.config["DDIM_T_sample"]
        sample_Ts = list(np.array(list(range(0, self.config["T"], skip))) + 1)
        sample_Ts_pre = [0] + sample_Ts[:-1]

        # init pure noises
        cur_x = torch.randn(shape_n).cuda()
        cur_e = torch.randn(shape_e).cuda()
        cur_net = (cur_x, cur_e)

        # denoising
        denoising_seq = [(cur_x.to(torch.device("cpu")), cur_e.to(torch.device("cpu")))]
        for t, t_pre in list(zip(reversed(sample_Ts), reversed(sample_Ts_pre))):
            # t of one denoising step
            t = torch.LongTensor([t]).cuda()
            t_pre = torch.LongTensor([t_pre]).cuda()
            # one step
            cur_x, cur_e = cur_net
            cur_e = cur_e * batchlization
            cur_net = (cur_x, cur_e)
            cur_net = self.DDIM_sample(cur_net, t, t_pre, c)
            # save
            cur_x, cur_e = cur_net
            cur_x, cur_e = cur_x.to(torch.device("cpu")), cur_e.to(torch.device("cpu"))
            cur_net_cpu = (cur_x, cur_e)
            denoising_seq.append(cur_net_cpu)
        return denoising_seq