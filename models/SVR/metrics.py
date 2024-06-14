from collections import defaultdict

import numpy as np
from scipy.stats import entropy



def cal_od_metrics(a, b):
    '''
    b has to be the groundtruth
    '''
    metrics = {
        "num_regions" : num_regions(a, b),
        "RMSE" : RMSE(a, b).item(),
        "NRMSE" : NRMSE(a, b).item(),
        "MAE" : MAE(a, b).item(),
        "MAPE" : MAPE(a, b).item(),
        "SMAPE" : SMAPE(a, b).item(),
        "CPC" : CPC(a, b).item(),

        "RMSE_nonzero" : RMSE_nonzero(a, b).item(),
        "MAE_nonzero": MAE_nonzero(a, b).item(),
        "MAPE_nonzero" : MAPE_nonzero(a, b).item(),
        "SMAPE_nonzero" : SMAPE_nonzero(a, b).item(),
        "CPC_nonzero" : CPC_nonzero(a, b).item(),

        "accuracy" : accuracy(a, b).item(),
        "matrix_COS_similarity" : matrix_COS_similarity(a, b).item(),
        "JSD_inflow" : JSD_inflow(a, b).item(),
        "JSD_outflow" : JSD_outflow(a, b).item(),
        "JSD_ODflow" : JSD_ODflow(a, b).item()
    }
    return metrics


def RMSE(a, b):
    if type(a) == type(np.array([1, 1])):
        return np.sqrt(((a - b) **2).mean())
    else:
        return ((a - b) **2).mean().sqrt()

def NRMSE(a, b):
    '''
    b has to be the groundtruth.
    '''
    return RMSE(a, b) / b.std()

def MAE(a, b):
    if type(a) == type(np.array([1, 1])):
        return np.abs(a - b).mean()
    else:
        return (a - b).abs().mean()

def MAPE(a, b):
    '''
    b has to be the groundtruth.
    '''
    if type(a) == type(np.array([1, 1])):
        return (np.abs(a - b) / (np.abs(b) + 1)).mean()
    else:
        return ((a - b).abs() / (b.abs() + 1)).mean()

def MSE(a, b):
    return ((a - b) **2).mean()

def SMAPE(a, b):
    if type(a) == type(np.array([1, 1])):
        return ( np.abs(a - b) / ( (np.abs(a) + np.abs(b)) / 2 + 1e-20) ).mean()
    else:
        return ( (a - b).abs() / ( (a.abs() + b.abs()) / 2 + 1e-20) ).mean()

def CPC(a, b):
    if ((a < 0).sum() + (b < 0).sum()) > 0:
        raise("OD flow should not be less than zero.")
    
    min = np.minimum(a, b)
    return 2 * min.sum() / ( a.sum() + b.sum())
    

def RMSE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return RMSE(a, b)

def MSE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MSE(a, b)

def NRMSE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return NRMSE(a, b)

def MAE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MAE(a, b)

def MAPE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return MAPE(a, b)

def SMAPE_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return SMAPE(a, b)

def CPC_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return CPC(a, b)

def accuracy(a, b):
    a[a<1] = 0
    idx_a = a.nonzero()
    idx_b = b.nonzero()
    
    a = np.zeros(a.shape)
    a[idx_a] = 1
    b = np.zeros(b.shape)
    b[idx_b] = 1
        
    sim = (a == b).sum() / (a.shape[0] **2)
    return sim

def matrix_COS_similarity(a, b):
    # row
    if type(a) == type(np.array([1, 1])):
        a_row_norm = np.sqrt((a **2).sum(0))
        b_row_norm = np.sqrt((b **2).sum(0))
    else:
        a_row_norm = (a **2).sum(0).sqrt()
        b_row_norm = (b **2).sum(0).sqrt()
    row_sim = (a * b).sum(0) / (a_row_norm * b_row_norm + 1e-20)

    # col
    if type(a) == type(np.array([1, 1])):
        a_col_norm = np.sqrt((a **2).sum(1))
        b_col_norm = np.sqrt((b **2).sum(1))
    else:
        a_col_norm = (a **2).sum(1).sqrt()
        b_col_norm = (b **2).sum(1).sqrt()
    col_sim = (a * b).sum(1) / (a_col_norm * b_col_norm + 1e-20)
    
    final_sim = (row_sim.sum() + col_sim.sum()) / (row_sim.shape[0] * 2)
    return final_sim

def values_to_bucket(values):
    # 2的指数分桶
    max_ = values.max()
    i = 0
    leftright = []
    nums = []
    while True:
        if i == 0:
            left = 0
            right = 1
            leftright.append(left)
            leftright.append(right)
            i += 1
        else:
            left = i
            right = i * 2
            leftright.append(right)
            i = i *2
        nums.append(((values >= left) & (values <right)).sum())

        if right > max_:
            break
    return leftright, nums

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * entropy(p, M, base=2) + 0.5 * entropy(q, M, base=2)

def JSD_in(a, b):
    """
    b should be the label.
    """
    a_in, b_in = a.sum(0), b.sum(0)
    sections, b_dist = values_to_bucket(b_in)
    a_dist = []
    for i in range(len(sections)-1):
        low, high = sections[i], sections[i+1]
        frequency = np.sum((a_in >= low) & (a_in < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    
    return JS_divergence(a_dist, b_dist)

def JSD_out(a, b):
    """
    b should be the label.
    """
    a_out, b_out = a.sum(1), b.sum(1)
    sections, b_dist = values_to_bucket(b_out)
    a_dist = []
    for i in range(len(sections)-1):
        low, high = sections[i], sections[i+1]
        frequency = np.sum((a_out >= low) & (a_out < high))
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    
    return JS_divergence(a_dist, b_dist)

def JSD_indegree(a, b):
    return JSD_in(a, b)

def JSD_outdegree(a, b):
    return JSD_out(a, b)

def JSD_inflow(a, b):
    return JSD_in(a, b)

def JSD_outflow(a, b):
    return JSD_out(a, b)

def JSD_ODflow(a, b):
    a, b = a.reshape([-1]), b.reshape([-1])
    sections, b_dist = values_to_bucket(b)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i+1]
        frequency = np.sum( (a >= low) & (a < high) )
        a_dist.append(frequency)
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()

    return JS_divergence(a_dist, b_dist)

def false_negative_rate(a, b):
    """
    b should be the label.
    """
    return np.sum((a == 0) & (b == 1)) / np.sum(b == 1)

def false_positive_rate(a, b):
    """
    b should be the label.
    """
    if np.sum(b==0) != 0:
        fpr = np.sum((a == 1) & (b == 0)) / np.sum(b == 0)
    else:
        fpr = np.float32(np.nan)
    return fpr

def nonzero_flow_fraction(a, b):
    """
    b should be the label.
    """
    a = (a == 1).sum() / (a.shape[0] **2)
    b = (b == 1).sum() / (b.shape[0] **2)
    d = np.float32(np.abs(a - b) / b)
    return d

def num_regions(a, b):
    return b.shape[0]

def average_listed_metrics(listed_metrics):
    sums = defaultdict(float)
    for d in listed_metrics:
        for key, value in d.items():
            sums[key] += value
    averages = {key: value / len(listed_metrics) for key, value in sums.items()}
    return averages

def citywise_segmented_metrics(valid_metrics):
    SEG_metrics = {
        "(0, 10]" : [],
        "(10, 50]" : [],
        "(50, 100]" : [],
        "(100, 200]" : [],
        "(200, 500]" : [],
        "(500, 1000]" : [],
        "(1000, 2000]" : [],
        "(2000, +inf]" : []
    }
    for item in valid_metrics:
        if 0 < item["num_regions"] <= 10:
            SEG_metrics["(0, 10]"].append(item)
        elif 10 < item["num_regions"] <= 50:
            SEG_metrics["(10, 50]"].append(item)
        elif 50 < item["num_regions"] <= 100:
            SEG_metrics["(50, 100]"].append(item)
        elif 100 < item["num_regions"] <= 200:
            SEG_metrics["(100, 200]"].append(item)
        elif 200 < item["num_regions"] <= 500:
            SEG_metrics["(200, 500]"].append(item)
        elif 500 < item["num_regions"] <= 1000:
            SEG_metrics["(500, 1000]"].append(item)
        elif 1000 < item["num_regions"] <= 2000:
            SEG_metrics["(1000, 2000]"].append(item)
        elif 2000 < item["num_regions"]:
            SEG_metrics["(2000, +inf]"].append(item)

    SEG_metrics["(0, 10]"] = average_listed_metrics(SEG_metrics["(0, 10]"])
    SEG_metrics["(10, 50]"] = average_listed_metrics(SEG_metrics["(10, 50]"])
    SEG_metrics["(50, 100]"] = average_listed_metrics(SEG_metrics["(50, 100]"])
    SEG_metrics["(100, 200]"] = average_listed_metrics(SEG_metrics["(100, 200]"])
    SEG_metrics["(200, 500]"] = average_listed_metrics(SEG_metrics["(200, 500]"])
    SEG_metrics["(500, 1000]"] = average_listed_metrics(SEG_metrics["(500, 1000]"])
    SEG_metrics["(1000, 2000]"] = average_listed_metrics(SEG_metrics["(1000, 2000]"])
    SEG_metrics["(2000, +inf]"] = average_listed_metrics(SEG_metrics["(2000, +inf]"])

    return SEG_metrics