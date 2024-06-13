import time

from sklearn.ensemble import RandomForestRegressor
from data_load import load_data
from metrics import *

from pprint import pprint


print("\n  **Loading data...")
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()

random_forest = RandomForestRegressor(n_estimators = 20,
                                      oob_score = True,
                                      max_depth = None,
                                      min_samples_split = 2,
                                      min_samples_leaf = 2,
                                      n_jobs = 20)

print('\n  **Start fitting...')
start = time.time()
random_forest.fit(xtrain, ytrain)

print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print("\n  **Evaluating...")
metrics_all = []
for x_one, y_one in zip(xtest, ytest):
    y_one_hat = random_forest.predict(x_one)
    y_one, y_one_hat = y_one.reshape([int(np.sqrt(x_one.shape[0])), 
                                      int(np.sqrt(x_one.shape[0]))]), y_one_hat.reshape([int(np.sqrt(x_one.shape[0])), 
                                                                                         int(np.sqrt(x_one.shape[0]))])
    metrics = cal_od_metrics(y_one_hat, y_one)
    metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)