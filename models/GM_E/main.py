import time

from data_load import load_data
from metrics import *
from model import *

import torch

from pprint import pprint


print("\n  **Loading data...")
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()

gravity = GRAVITY()
gravity = gravity.cuda()

print('\n  **Start fitting...')
start = time.time()

xtrain = torch.FloatTensor(xtrain).cuda()
ytrain = torch.FloatTensor(ytrain).cuda()

optimizer = torch.optim.Adam(gravity.parameters(), lr=1e-4)


best_valid_loss = np.inf
valid_flag = 100
for i in range(10000):
    print(f"Epoch {i+1}:", end=" | ")
    gravity.train()

    optimizer.zero_grad()

    yhat = gravity(xtrain).squeeze()
    loss = torch.mean((yhat-ytrain)**2)
    loss.backward()
    optimizer.step()
    loss_value = loss.item()
    print(f"train loss={loss_value:.7g}", end=" | ")

    valid_losses = []
    for xvalid_one, yvalid_one in zip(xvalid, yvalid):
        xvalid_one = torch.FloatTensor(xvalid_one).cuda()
        yvalid_one = torch.FloatTensor(yvalid_one).cuda()
        gravity.eval()
        yhat = gravity(xvalid_one).squeeze()
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
metrics_all = []
for x_one, y_one in zip(xtest, ytest):
    x_one = torch.FloatTensor(x_one).cuda()
    gravity.eval()
    y_one_hat = gravity(x_one).squeeze()
    y_one_hat = y_one_hat.cpu().detach().numpy()

    y_one_hat = y_one_hat.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])
    y_one = y_one.reshape([int(np.sqrt(y_one.shape[0])), int(np.sqrt(y_one.shape[0]))])

    metrics = cal_od_metrics(y_one_hat, y_one)
    metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)