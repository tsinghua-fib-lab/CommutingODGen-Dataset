import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from pprint import pprint

from utils.tool import *
from utils.metrics import *





def train(config, diff_model, train_Dloader, valid_Dloader, logger, scalers):
    optm = torch.optim.AdamW(diff_model.model.parameters(), lr=config["learning_rate"], weight_decay=0.001)
    scheduler = StepLR(optm, step_size=5000, gamma=0.9)

    start_epochs = 0
    Ts = [x for x in range(config["T"])]
    for epoch in range(start_epochs, config["EPOCH"]):
        ###################### train
        diff_model.train()
        print("  ** Training. Epoch:", epoch)
        train_loss = 0
        for i, batch in tqdm(enumerate(train_Dloader), total=len(train_Dloader), unit="batch"):
            _, batched_graph, batched_dis, ods = batch

            optm.zero_grad()

            # t
            t = torch.LongTensor([random.choice(Ts) for _ in ods]).cuda()
            # n
            n = batched_graph.ndata["demo"].cuda()
            # e
            dim = sum([x.shape[0] for x in ods])
            batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
            batchlization = torch.full((dim, dim), 0, dtype=torch.float32).cuda()
            l, r = 0, 0
            for i, od in enumerate(ods):
                l = r
                r = r + od.shape[0]
                batched_od[l:r, l:r] = ods[i]
                batchlization[l:r, l:r] = 1
            e = batched_od.cuda()
            # net
            net = (n, e)
            # mask
            mask_n, mask_e = generate_masks(config, n, e)
            mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
            masks = (mask_n, mask_e)
            # loss
            loss = diff_model.loss(net, masks, t, batched_dis.cuda(), batchlization)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
            optm.step()
        scheduler.step()
        train_loss = train_loss / len(train_Dloader)
        print("--------------------------------------------------")

        ###################### valid
        diff_model.eval()
        if (epoch % config["valid_period"] == 0) and (epoch != 0):
            print("--------------------------------------------------")
            with torch.no_grad():
                print("  ** Valid. Epoch:", epoch)
                valid_evaluator = validEvaluator(config)
                valid_loss = 0
                for i, batch in tqdm(enumerate(valid_Dloader), total=len(valid_Dloader), unit="batch"):
                    _, batched_graph, batched_dis, ods = batch

                    # t
                    t = torch.LongTensor([random.choice(Ts)]).cuda()
                    # n
                    n = batched_graph.ndata["demo"].cuda()
                    # e
                    dim = sum([x.shape[0] for x in ods])
                    batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                    batchlization = torch.full((dim, dim), 0, dtype=torch.float32).cuda()
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
                    # mask
                    mask_n, mask_e = generate_masks(config, n, e)
                    mask_n, mask_e = mask_n.cuda(), mask_e.cuda()
                    masks = (mask_n, mask_e)

                    # loss
                    loss = diff_model.loss(net, masks, t, batched_dis, batchlization)
                    valid_loss += loss.item()

                    # metrics
                    mask_n, mask_e, flags = generate_eval_mask(config, n, e)
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

                    # eval
                    pred = (n_hat.cpu().numpy(), e_hat.cpu().numpy())
                    masks = (mask_n.cpu().numpy(), mask_e.cpu().numpy())
                    valid_evaluator.evaluate_batch(pred=pred,
                                                   gt=(n, ods),
                                                   masks=masks,
                                                   flags=flags,
                                                   scalers=scalers)
                valid_loss = valid_loss / len(valid_Dloader)
                metrics = valid_evaluator.summary_all_metrics()

                logger.once_valid_record(epoch, valid_loss, metrics, diff_model)

            print("--------------------------------------------------")
            if logger.overfit_flag >= config["overfit_tolerance"]:
                print(" ** Early stop!")
                print("--------------------------------------------------")
                break