import time

from sklearn.ensemble import GradientBoostingRegressor

from data_load import *
from metrics import *
from model import *


from pprint import pprint

print("\n  **Loading data...")
nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, nfeats_test, adjs_test, dises_test, ods_test = load_data()

nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)

gmel = GMEL()
gmel = gmel.cuda()

optimizer = torch.optim.Adam(gmel.parameters(), lr=3e-4)

print('\n  **Start fitting...')
start = time.time()

best_valid_loss = np.inf
valid_flag = 10
for i in range(1000):
    start_epoch = time.time()
    print(f"Epoch {i+1}:", end=" | ")
    gmel.train()

    loss_epoch = []
    for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
        nfeat = nfeat_scaler.transform(nfeat)
        dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
        
        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))
        dis = torch.FloatTensor(dis).cuda()
        od = torch.FloatTensor(od).cuda()

        optimizer.zero_grad()

        flow_in, flow_out, flow, h_in, h_out = gmel(g, nfeat)
        loss = torch.mean((flow_in-od.sum(0))**2) + torch.mean((flow_out-od.sum(1))**2) + torch.mean((flow-od)**2)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_epoch.append(loss_value)
    loss_value = np.mean(loss_epoch)
    print(f"train loss={loss_value:.7g}", end=" | ")

    with torch.no_grad():
        valid_losses = []
        for nfeat, adj, dis, od in zip(nfeats_valid, adjs_valid, dises_valid, ods_valid):
            nfeat = nfeat_scaler.transform(nfeat)
            dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
            od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)
            
            nfeat = torch.FloatTensor(nfeat).cuda()
            g = build_graph(adj).to(torch.device('cuda'))
            dis = torch.FloatTensor(dis).cuda()
            od = torch.FloatTensor(od).cuda()

            flow_in, flow_out, flow, h_in, h_out = gmel(g, nfeat)
            loss = torch.mean((flow_in-od.sum(0))**2) + torch.mean((flow_out-od.sum(1))**2) + torch.mean((flow-od)**2)
            valid_losses.append(loss.item())
        valid_loss = np.mean(valid_losses)
        print(f"valid loss={valid_loss:.7g}", end=" | ")
        print(f"consume {time.time()-start_epoch:.2f} seconds")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            valid_flag = 10
        else:
            valid_flag -= 1
            if valid_flag == 0:
                break

print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

gbrt = GradientBoostingRegressor(n_estimators=20,
                                 min_samples_split=2,
                                 min_samples_leaf=2,
                                 max_depth=None)

print("\n  **Training GBRT...")
start = time.time()

xtrain = []
ytrain = []
with torch.no_grad():
    for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
        nfeat = nfeat_scaler.transform(nfeat)
        
        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))

        _, _, _, h_in, h_out = gmel(g, nfeat)
        h_in = h_in.cpu().detach().numpy()
        h_out = h_out.cpu().detach().numpy()
        feat = np.concatenate([h_in, h_out], axis=1)

        feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
        feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
        feat = np.concatenate([feat_o, feat_d, dis.reshape([dis.shape[0], dis.shape[0], 1])], axis=2).reshape([-1, feat.shape[1]*2+1])
        xtrain.append(feat)
        ytrain.append(od.reshape(-1))
        
xtrain = np.concatenate(xtrain, axis=0)
ytrain = np.concatenate(ytrain, axis=0)

gbrt.fit(xtrain, ytrain)

print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print("\n  **Evaluating...")
metrics_all = []
for nfeat, adj, dis, od in zip(nfeats_test, adjs_test, dises_test, ods_test):
    nfeat = nfeat_scaler.transform(nfeat)
    nfeat = torch.FloatTensor(nfeat).cuda()
    g = build_graph(adj).to(torch.device('cuda'))

    with torch.no_grad():
        _, _, _, h_in, h_out = gmel(g, nfeat)
        h_in = h_in.cpu().detach().numpy()
        h_out = h_out.cpu().detach().numpy()
        feat = np.concatenate([h_in, h_out], axis=1)

        feat_o = feat.reshape([feat.shape[0], 1, feat.shape[1]]).repeat(feat.shape[0], axis=1)
        feat_d = feat.reshape([1, feat.shape[0], feat.shape[1]]).repeat(feat.shape[0], axis=0)
        feat = np.concatenate([feat_o, feat_d, dis.reshape([dis.shape[0], dis.shape[0], 1])], axis=2).reshape([-1, feat.shape[1]*2+1])

        od_hat = gbrt.predict(feat).reshape([int(np.sqrt(feat.shape[0])), int(np.sqrt(feat.shape[0]))])
        od = od.reshape([int(np.sqrt(feat.shape[0])), int(np.sqrt(feat.shape[0]))])
        metrics = cal_od_metrics(od_hat, od)
        metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)