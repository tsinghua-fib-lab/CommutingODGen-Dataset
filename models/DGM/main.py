import time

from sklearn.preprocessing import MinMaxScaler
from data_load import load_data
from metrics import *
from model import *

from torch.utils.data import DataLoader, TensorDataset

from pprint import pprint


print("\n  **Loading data...")
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()
odmin_, odmax_ = ytrain.min(), ytrain.max()

feat_scaler = MinMaxScaler((-1, 1)).fit(xtrain)
od_scaler = OD_normer(ytrain.min(), ytrain.max())

deepgravity = DeepGravity()
deepgravity = deepgravity.cuda()

xtrain = torch.FloatTensor(feat_scaler.transform(xtrain))
ytrain = torch.FloatTensor(od_scaler.normalize(ytrain))

ds = TensorDataset(xtrain, ytrain)
dl = DataLoader(ds, batch_size=1000000, shuffle=True)

optimizer = torch.optim.Adam(deepgravity.parameters(), lr=3e-4)

print('\n  **Start fitting...')
start = time.time()

best_valid_loss = np.inf
valid_flag = 100
for i in range(10000):
    print(f"Epoch {i+1}:", end=" | ")
    deepgravity.train()

    loss_epoch = []
    for xbatch, ybatch in dl:
        xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
        optimizer.zero_grad()

        yhat = deepgravity(xbatch).squeeze()
        loss = torch.mean((yhat-ybatch)**2)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_epoch.append(loss_value)
    loss_value = np.mean(loss_epoch)
    print(f"train loss={loss_value:.7g}", end=" | ")

    with torch.no_grad():
        valid_losses = []
        for xvalid_one, yvalid_one in zip(xvalid, yvalid):
            xvalid_one = feat_scaler.transform(xvalid_one)
            yvalid_one = od_scaler.normalize(yvalid_one)
            xvalid_one = torch.FloatTensor(feat_scaler.transform(xvalid_one)).cuda()
            yvalid_one = torch.FloatTensor(od_scaler.normalize(yvalid_one)).cuda()
            deepgravity.eval()
            yhat = deepgravity(xvalid_one).squeeze()
            yhat = yhat.cpu().detach().numpy()
            valid_loss = (yhat-yvalid_one.cpu().numpy())**2
            valid_losses.append(valid_loss)
        valid_loss = np.concatenate(valid_losses).mean()
        print(f"valid loss={valid_loss:.7g}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            valid_flag = 100
        else:
            valid_flag -= 1
            if valid_flag == 0:
                print('Early stopping!')
                break

print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print("\n  **Evaluating...")
with torch.no_grad():
    metrics_all = []
    for x_one, y_one in zip(xtest, ytest):
        x_one = feat_scaler.transform(x_one)
        x_one = torch.FloatTensor(x_one).cuda()
        deepgravity.eval()
        y_one_hat = deepgravity(x_one).squeeze()
        y_one_hat = y_one_hat.cpu().detach().numpy()

        y_one_hat = od_scaler.renormalize(y_one_hat)

        y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
        y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
        y_one_hat[y_one_hat < 0] = 0

        metrics = cal_od_metrics(y_one_hat, y_one)
        metrics_all.append(metrics)

    avg_metrics = average_listed_metrics(metrics_all)
    pprint(avg_metrics)