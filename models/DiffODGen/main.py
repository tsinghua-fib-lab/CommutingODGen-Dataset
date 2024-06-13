import time
from random import shuffle

from utils.procedure import *
from data_load import *
from model import *
from utils.MyLogger import Logger
from metrics import *



config = {
    "data_path" : "data/",
    "shuffle_cities" : 0,

    "attr_MinMax" : 1,
    "od_MinMax" : 1,
    "skew_norm" : "log",
    
    "topo_sample_step" : "digress",
    "topo_train_objective" : "x_0",

    "sample_method" : "DDIM",
    "teacher_force" : 1,

    "topo_diffusion" : 1,
    "flow_diffusion" : 0,

    "topo_train" : 1,
    "flow_train" : 1,

    "train_set": 0.8,
    "valid_set": 0.1,
    "test_set": 0.1,

    "T_topo": 1000,
    "T_flow": 1000,
    "DDIM_T_sample" : 100,
    "DDIM_eta" : 0, 
    "beta_scheduler_topo" : "cosine",
    "beta_scheduler_flow" : "cosine",

    "Topo_e_classes" : 2,
    "Topo_e_weight" : 0.5,

    "NN_type" : "GTN",

    "topo_Cemb" : "mlp",

    "GAT_layers" : 3,
    "GAT_indim" : 131,
    "GAT_hiddim" : 64,
    "GAT_outdim" : 32,
    "GAT_heads" : 6,
    "GAT_feat_drop" : 0,
    "GAT_attn_drop" : 0,
    "GAT_negative_slope" : 0,
    "GAT_residual" : 1,

    "disProj_layers" : 4,
    "disProj_hiddim" : 64,
    
    "T_GTN_x_indim" : 8,
    "T_GTN_x_hiddim" : 32,
    "T_GTN_dim_ffX" : 32,
    "T_GTN_x_outdim" : 1,
    
    "T_GTN_e_indim" : 2,
    "T_GTN_e_hiddim" : 32,
    "T_GTN_dim_ffE" : 32,
    "T_GTN_e_outdim" : 2,


    "GTN_x_indim" : 8,
    "GTN_x_hiddim" : 64,
    "GTN_dim_ffX" : 64,
    "GTN_x_outdim" : 1,

    "GTN_e_indim" : 1,
    "GTN_e_hiddim" : 64,
    "GTN_dim_ffE" : 64,
    "GTN_e_outdim" : 1,

    "GTN_n_head" : 2,
    "GTN_dropout" : 0,


    "diffFusion_layers" : 4,
    "diffFusion_dim" :64,

    "converge_check" : 0,

    "EPOCH" : 10000,
    "topo_num_t_epoch" : 100,
    "flow_num_t_epoch" : 1000,
    "lr_topo": 3e-4,
    "lr_flow": 2e-5,
    "optm" : "AdamW"
}

print("\n  **Loading data...")
nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, nfeats_test, adjs_test, dises_test, ods_test = load_data()
nfeats_train, adjs_train, dises_train, ods_train = nfeats_train[:1], adjs_train[:1], dises_train[:1], ods_train[:1]

nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)


# logger
logger = Logger(config)

# model
model_T = TopoDiffusion(config)
model_F = FlowDiffusion(config)
model_T = model_T.cuda()
model_F = model_F.cuda()

# optimizer
optimizer_T = torch.optim.Adam(model_T.NN.parameters(), lr=config["lr_topo"])
optimizer_F = torch.optim.Adam(model_F.NN.parameters(), lr=config["lr_flow"])

print('\n  **Start fitting topo...')
start = time.time()
for i in range(10000):
    start_epoch = time.time()
    print(f"Epoch {i+1}:", end=" | ")
    model_T.train()
    model_F.train()

    loss_epoch = []
    for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
        nfeat = nfeat_scaler.transform(nfeat)
        dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
        
        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))
        dis = torch.FloatTensor(dis).cuda()
        od = torch.FloatTensor(od).cuda()

        topo = torch.zeros_like(od).cuda()
        topo[od.nonzero()] = 1

        condition = (nfeat, dis, g, topo)

        clean_topo = topo.clone()
        clean_topo = F.one_hot(clean_topo.long(), num_classes=2).float()

        Ts = [x for x in range(config["T_topo"])]
        shuffle(Ts)
        Ts = Ts[:config["topo_num_t_epoch"]]
        
        t = random.choice(Ts)
    
        optimizer_T.zero_grad()
        t = torch.LongTensor([t])
        loss = model_T.loss(clean_topo, t, condition)

        loss_value = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_T.parameters(), 1.)
        optimizer_T.step()

        loss_epoch.append(loss_value)

    loss_value = np.mean(loss_epoch)
    print(f"train loss={loss_value:.7g}")


print('\n  **Start fitting flow...')
start = time.time()
loss_epoch = []
for i in range(10000):
    start_epoch = time.time()
    print(f"Epoch {i+1}:", end=" | ")
    model_F.train()

    loss_epoch = []
    for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
        nfeat = nfeat_scaler.transform(nfeat)
        dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
        
        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))
        dis = torch.FloatTensor(dis).cuda()
        od = torch.FloatTensor(od).cuda()

        topo = torch.zeros_like(od).cuda()
        topo[od.nonzero()] = 1


        with torch.no_grad():
            condition = (nfeat, dis, g, topo)
            topo_gen = model_T.sample(topo.shape, condition).cuda()

        topo_train = torch.zeros_like(topo).cuda()
        topo_train[(topo == 1) | (topo_gen == 1)] = 1

        condition = (nfeat, dis, g, topo_train)
        Ts = [x for x in range(config["T_flow"])]
        shuffle(Ts)
        Ts = Ts[:config["flow_num_t_epoch"]]
        t = random.choice(Ts)

        optimizer_F.zero_grad()
        t = torch.LongTensor([t])
        loss = model_F.loss(od, t, condition)

        loss_value = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_F.parameters(), 1.)
        optimizer_F.step()

        loss_epoch.append(loss_value)
    loss_value = np.mean(loss_epoch)
    print(f"train loss={loss_value:.7g}")


print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print('\n  **Start testing...')

model_T.eval()
model_F.eval()
metrics_all = []
with torch.no_grad():
    for nfeat, adj, dis, od in zip(nfeats_test, adjs_test, dises_test, ods_test):
        nfeat = nfeat_scaler.transform(nfeat)
        dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        od_real = od
        od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
        
        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))
        dis = torch.FloatTensor(dis).cuda()
        od = torch.FloatTensor(od).cuda()

        condition = (nfeat, dis, g, od)
        topo_gen = model_T.sample(od.shape, condition).cuda()

        condition = (nfeat, dis, g, topo_gen)
        od_hat = model_F.DDIM_sample_loop(od.shape, condition)[-1]

        od_hat = od_scaler.inverse_transform(od_hat.cpu().numpy().reshape(-1, 1)).reshape(od.shape)
        od_hat[od_hat < 0] = 0

        metrics = cal_od_metrics(od_hat, od_real)
        metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)
