import os
import warnings
warnings.filterwarnings("ignore")

import torch

from tqdm import tqdm
from pprint import pprint

from utils.tool import generate_Gen_mask, generate_Comp_mask, generate_MissGen_mask, generate_MissComp_mask, mean_the_denoising_process
from utils.metrics import *


def test(config, diff_model, test_Dloader, logger, scalers):

    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Evaluation. :")
        Gen_metrics = []
        Comp_metrics = []
        feat_metrics = []
        MissGen_od_metrics = []
        MissComp_od_metrics = []
        for i, batch in tqdm(enumerate(test_Dloader), total=len(test_Dloader)):
            geoids, batched_graph, batched_dis, ods = batch
            # n
            n = batched_graph.ndata["demo"].cuda()
            # e
            dim = sum([x.shape[0] for x in ods])
            batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
            batchlization = torch.full((dim, dim), 0, dtype=torch.float32).cuda() # There are zero-noise from blocking the OD matrix.
            l, r = 0, 0
            for i, od in enumerate(ods):
                l = r
                r = r + od.shape[0]
                batched_od[l:r, l:r] = ods[i]
                batchlization[l:r, l:r] = 1
            e = batched_od.cuda()
            # net
            net = (n, e)
            # dis
            batched_dis = batched_dis.cuda()

            Gen_metrics.extend(eval_gen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers))
        
        # summary metrics and save log
        eval_metrics = summary_all_metrics(Gen_metrics if len(Gen_metrics) > 0 else None)
        pprint(eval_metrics)

    print("--------------------------------------------------")
    
def test_one_city(config, city, scalers, diff_model):
    # load data
    nfeat = np.load(config["data_path"] + city + "/nfeat/nfeat.npy")
    dis = np.load(config["data_path"] + city + "/adj/dis.npy")
    od = np.load(config["data_path"] + city + "/od/od.npy")

    # normalization
    nfeat = scalers["feat"].transform(nfeat)
    dis = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    # Tensor
    nfeat = torch.from_numpy(nfeat).float().cuda()
    dis = torch.from_numpy(dis).float().cuda()

    # generation
    


def summary_all_metrics(Gen_metrics=None):
    gen = average_listed_metrics(Gen_metrics)

    return gen


def eval_gen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers):
    diff_model.eval()
    n, e = net
    mask_n, mask_e = generate_Gen_mask(n, e)
    mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
    masks = (mask_n, mask_e)
    c = (net, masks, batched_dis, batchlization)

    # generate
    n_hats, e_hats = [], []
    denoising_seqs = []
    for _ in range(config["sample_times"]):
        denoising_seq = diff_model.DDIM_sample_loop(n.shape, e.shape, c)
        denoising_seqs.append(denoising_seq)
        net_hat = denoising_seq[-1]
        n_hat, e_hat = net_hat
        n_hats.append(n_hat)
        e_hats.append(e_hat)
    denoising_seq = mean_the_denoising_process(denoising_seqs)
    n_hat = torch.mean(torch.stack(n_hats), dim=0)
    e_hat = torch.mean(torch.stack(e_hats), dim=0)
    n_hat, e_hat = n_hat.cpu().numpy(), e_hat.cpu().numpy()


    # eval
    mask_n, mask_e = mask_n.cpu().numpy(), mask_e.cpu().numpy()
    all_metrics = []
    l, r = 0, 0
    for idx, od in enumerate(ods):
        l = r
        r = r + od.shape[0]
        # extract
        od_hat = e_hat[l:r, l:r]
        tmp_mask_e = mask_e[l:r, l:r]

        if np.sum(tmp_mask_e == 0) == 0:
            continue
        
        od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        od = scalers["od_normer"].inverse_transform(od)
        od[od < 0] = 0
        od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
        od_hat = scalers["od_normer"].inverse_transform(od_hat)
        for i in range(od_hat.shape[0]): # consistant with groundtruth
            od_hat[i,i] = 0
        od_hat[od_hat < 0] = 0
        od_hat = np.floor(od_hat)

        one_metrics = cal_od_metrics(od_hat, od)
        all_metrics.append(one_metrics)

    return all_metrics


def eval_comp(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers):
    diff_model.eval()
    n, e = net
    mask_n, mask_e = generate_Comp_mask(n, e)
    mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
    masks = (mask_n, mask_e)
    c = (net, masks, batched_dis, batchlization)

    # generate
    n_hats, e_hats = [], []
    for _ in range(config["sample_times"]):
        net_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
        n_hat, e_hat = net_hat
        n_hats.append(n_hat)
        e_hats.append(e_hat)
    n_hat = torch.mean(torch.stack(n_hats), dim=0)
    e_hat = torch.mean(torch.stack(e_hats), dim=0)
    n_hat, e_hat = n_hat.cpu().numpy(), e_hat.cpu().numpy()

    # eval
    mask_n, mask_e = mask_n.cpu().numpy(), mask_e.cpu().numpy()
    all_metrics = []
    l, r = 0, 0
    for _, od in enumerate(ods):
        l = r
        r = r + od.shape[0]
        # extract
        od_hat = e_hat[l:r, l:r]
        tmp_mask_e = mask_e[l:r, l:r]

        if np.sum(tmp_mask_e == 0) == 0:
            continue
        
        od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        od = scalers["od_normer"].inverse_transform(od)
        od[od < 0] = 0
        od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
        od_hat = scalers["od_normer"].inverse_transform(od_hat)
        for i in range(od_hat.shape[0]): # consistant with groundtruth
            od_hat[i,i] = 0
        od_hat[od_hat < 0] = 0
        od_hat = np.floor(od_hat)

        od = od[tmp_mask_e == 0]
        od_hat = od_hat[tmp_mask_e == 0]

        one_metrics = cal_NonMatrix_metrics(od_hat, od)
        all_metrics.append(one_metrics)
    
    return all_metrics


def eval_missgen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers):
    diff_model.eval()
    n, e = net
    mask_n, mask_e = generate_MissGen_mask(n, e)
    mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
    masks = (mask_n, mask_e)
    c = (net, masks, batched_dis, batchlization)

    # generate
    net_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
    n_hat, e_hat = net_hat
    n_hat, e_hat = n_hat.cpu().numpy(), e_hat.cpu().numpy()

    # eval
    mask_n, mask_e = mask_n.cpu().numpy(), mask_e.cpu().numpy()
    feat_metrics = []
    od_metrics = []
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

        nfeat = scalers["feat"].inverse_transform(nfeat.cpu().numpy())
        nfeat[nfeat < 0] = 0

        nfeat_hat = scalers["feat"].inverse_transform(nfeat_hat)
        nfeat_hat[nfeat_hat < 0] = 0
        nfeat_hat[tmp_mask_n != 0] = nfeat[tmp_mask_n != 0]

        one_feat_metrics = cal_feat_metrics(nfeat_hat, nfeat)
        feat_metrics.append(one_feat_metrics)

        # od
        if np.sum(tmp_mask_e == 0) == 0:
            continue

        od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        od = scalers["od_normer"].inverse_transform(od)
        od[od < 0] = 0
        od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
        od_hat = scalers["od_normer"].inverse_transform(od_hat)
        for i in range(od_hat.shape[0]): # consistant with groundtruth
            od_hat[i,i] = 0
        od_hat[od_hat < 0] = 0
        od_hat = np.floor(od_hat)

        one_od_metrics = cal_od_metrics(od_hat, od)
        od_metrics.append(one_od_metrics)
    
    return feat_metrics, od_metrics


def eval_misscomp(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers):
    os.makedirs(logger.generation_directory + "MISSCOMP/od", exist_ok=True)
    os.makedirs(logger.generation_directory + "MISSCOMP/feat", exist_ok=True)

    diff_model.eval()
    n, e = net
    mask_n, mask_e = generate_MissComp_mask(n, e)
    mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
    masks = (mask_n, mask_e)
    c = (net, masks, batched_dis, batchlization)

    # generate
    net_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
    n_hat, e_hat = net_hat
    n_hat, e_hat = n_hat.cpu().numpy(), e_hat.cpu().numpy()

    # eval
    mask_n, mask_e = mask_n.cpu().numpy(), mask_e.cpu().numpy()
    od_metrics = []
    l, r = 0, 0
    for _, od in enumerate(ods):
        l = r
        r = r + od.shape[0]
        # extract
        od_hat = e_hat[l:r, l:r]
        tmp_mask_e = mask_e[l:r, l:r]

        # od
        if np.sum(tmp_mask_e == 0) == 0:
            continue

        od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        od = scalers["od_normer"].inverse_transform(od)
        od[od < 0] = 0
        od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
        od_hat = scalers["od_normer"].inverse_transform(od_hat)
        for i in range(od_hat.shape[0]): # consistant with groundtruth
            od_hat[i,i] = 0
        od_hat[od_hat < 0] = 0
        od_hat = np.floor(od_hat)
        od = od[tmp_mask_e == 0]
        od_hat = od_hat[tmp_mask_e == 0]

        one_od_metrics = cal_NonMatrix_metrics(od_hat, od)
        od_metrics.append(one_od_metrics)
        
    return None, od_metrics


