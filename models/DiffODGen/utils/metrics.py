import numpy as np

import torch

import matplotlib.pyplot as plt
from scipy.stats import entropy





def cal_all_metrics_topo(a, b):
    '''
    b has to be the groundtruth
    '''
    metrics = {
        "CPC_topo" : CPC(a, b).item(),
        "false_positive_rate" : false_positive_rate(a, b).item(),
        "false_negative_rate" : false_negative_rate(a, b).item(),
        "accuracy" : accuracy(a, b).item(),
        "nonzero_flow_fraction" : nonzero_flow_fraction(a, b).item(),
        "JSD_indegree" : JSD_indegree(a, b).item(),
        "JSD_outdegree" : JSD_outdegree(a, b).item()
    }
    return metrics

def cal_all_metrics_flow(a, b):
    '''
    b has to be the groundtruth
    '''
    metrics = {
        "RMSE" : RMSE(a, b).item(),
        "NRMSE" : NRMSE(a, b).item(),
        "MAE" : MAE(a, b).item(),
        "MAPE" : MAPE(a, b).item(),
        # "MSE" : MSE(a, b).item(),
        "SMAPE" : SMAPE(a, b).item(),
        "CPC" : CPC(a, b).item(),


        "RMSE_nonzero" : RMSE_nonzero(a, b).item(),
        # "NRMSE_nonzero" : NRMSE_nonzero(a, b).item(),
        "MAE_nonzero": MAE_nonzero(a, b).item(),
        "MAPE_nonzero" : MAPE_nonzero(a, b).item(),
        # "MSE_nonzero" : MSE_nonzero(a, b).item(),
        "SMAPE_nonzero" : SMAPE_nonzero(a, b).item(),
        # "CPC_nonzero" : CPC_nonzero(a, b).item(),

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
    if type(a) == type(np.array([1, 1])):
        min = np.minimum(a, b)
        return 2 * min.sum() / ( a.sum() + b.sum())
    else:
        min = torch.minimum(a, b)
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
    if type(a) == type(np.array([1, 1])):
        a = np.zeros(a.shape)
        a[idx_a] = 1
        b = np.zeros(b.shape)
        b[idx_b] = 1
    else:
        idx_a = (idx_a[:, 0], idx_a[:, 1])
        idx_b = (idx_b[:, 0], idx_b[:, 1])
        a = torch.zeros_like(a)
        a[idx_a] = 1
        b = torch.zeros_like(b)
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
    if type(a) == type(np.array([1, 1])):
        a = torch.FloatTensor(a)
    if type(b) == type(np.array([1, 1])):
        b = torch.FloatTensor(b)
    a_in, b_in = a.sum(0).cpu().numpy(), b.sum(0).cpu().numpy()
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
    if type(a) == type(np.array([1, 1])):
        a = torch.FloatTensor(a)
    if type(b) == type(np.array([1, 1])):
        b = torch.FloatTensor(b)
    a_out, b_out = a.sum(1).cpu().numpy(), b.sum(1).cpu().numpy()
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
    if type(a) == type(np.array([1, 1])):
        a = torch.FloatTensor(a)
    if type(b) == type(np.array([1, 1])):
        b = torch.FloatTensor(b)
    a, b = a.cpu().numpy().reshape([-1]), b.cpu().numpy().reshape([-1])
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
    if type(a) == type(np.array([1, 1])):
        pass
    else:
        a = a.cpu().numpy()
    if type(b) == type(np.array([1, 1])):
        pass
    else:
        b = b.cpu().numpy()

    return np.sum((a == 0) & (b == 1)) / np.sum(b == 1)

def false_positive_rate(a, b):
    """
    b should be the label.
    """
    if type(a) == type(np.array([1, 1])):
        pass
    else:
        a = a.cpu().numpy()
    if type(b) == type(np.array([1, 1])):
        pass
    else:
        b = b.cpu().numpy()

    return np.sum((a == 1) & (b == 0)) / np.sum(b == 0)

def nonzero_flow_fraction(a, b):
    """
    b should be the label.
    """
    if type(a) == type(np.array([1, 1])):
        pass
    else:
        a = a.cpu().numpy()
    if type(b) == type(np.array([1, 1])):
        pass
    else:
        b = b.cpu().numpy()

    return (a == 1).sum() / (a.shape[0] **2)