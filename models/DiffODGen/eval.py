import torch
from utils.tool import build_DGLGraph, rescale_od, ToThTensor
from utils.metrics import cal_all_metrics

def ToBinary(mat):
    mat[mat > 1] = 1
    mat[mat != 1] = 0
    return mat

def check_condition_flow(config, model, shape, condition, data, logger, epoch):
    od = ToThTensor(data["src"]["od"])
    x_0 = od.to(config["devices"][0])
    valid_x_0 = rescale_od(x_0.to(torch.device("cpu")), data["src"]["od_min_max"])
    noise_levels = [0.1, 0.2, 0.4, 0.6, 0.9, 1]
    for noise_level in noise_levels:
        attr, dis, g, od_topo = condition

        attr = (1 - noise_level) * attr + noise_level * torch.rand_like(attr)

        dis = (1 - noise_level) * dis + noise_level * torch.rand_like(dis)

        adj = torch.zeros_like(od_topo)
        adj[g.edges()] = 1
        adj = (1 - noise_level) * adj + noise_level * torch.rand_like(adj)
        adj = ToBinary(adj).numpy()
        g = build_DGLGraph(adj)

        # od_topo = (1 - noise_level) * od_topo + noise_level * torch.rand_like(od_topo)
        # od_topo = ToBinary(od_topo)

        condition = (attr, dis, g, od_topo)
        x_seq = model.DDIM_sample_loop(shape, condition)
        x_0_hat = rescale_od(x_seq[-1], data["src"]["od_min_max"])
        topo = condition[3]
        zero_flow_idx = topo == 0
        x_0_hat[zero_flow_idx] = -1 # Zero flows are normalized to -1.
        x_0_hat = torch.clip(x_0_hat, min=0)
        all_metrics = cal_all_metrics(x_0_hat, valid_x_0)
        cpc = all_metrics["CPC"]
        logger.summary_writer.add_scalar("Check_C/noise_"+str(noise_level), cpc, epoch)


def check_condition_topo(config, model, shape, condition, data, logger, epoch):
    tar_cities = config["tar_cities"]
    