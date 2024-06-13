import os
import random

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch, dgl

def load_all_areas(if_shuffle=True):
    areas = os.listdir("data")
    if if_shuffle:
        random.shuffle(areas)
    return areas

def split_train_valid_test(areas, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    assert train_ratio + valid_ratio + test_ratio == 1

    train_areas = areas[:int(len(areas)*train_ratio)]
    valid_areas = areas[int(len(areas)*train_ratio):int(len(areas)*(train_ratio+valid_ratio))]
    test_areas = areas[int(len(areas)*(train_ratio+valid_ratio)):]
    return train_areas, valid_areas, test_areas


def construct_sample(areas):
    
    nfeats = []
    adjs = []
    dises = []
    ods = []
    for area in areas:
        demos = np.load(f"data/{area}/demos.npy")
        pois = np.load(f"data/{area}/pois.npy")
        nfeat = np.concatenate([demos, pois], axis=1)

        adj = np.load(f"data/{area}/adj.npy")

        dis = np.load(f"data/{area}/dis.npy")

        od = np.load(f"data/{area}/od.npy")

        nfeats.append(nfeat)
        adjs.append(adj)
        dises.append(dis)
        ods.append(od)

    return nfeats, adjs, dises, ods
    
def load_data(if_shuffle=True):
    areas = load_all_areas(if_shuffle)
    train_areas, valid_areas, test_areas = split_train_valid_test(areas)

    nfeats_train, adjs_train, dises_train, ods_train = construct_sample(train_areas)
    nfeats_valid, adjs_valid, dises_valid, ods_valid = construct_sample(valid_areas)
    nfeats_test, adjs_test, dises_test, ods_test = construct_sample(test_areas)

    return nfeats_train, adjs_train, dises_train, ods_train, \
           nfeats_valid, adjs_valid, dises_valid, ods_valid, \
           nfeats_test, adjs_test, dises_test, ods_test

def get_scalers(nfeats, dises, ods):
    nfeat_scaler = MinMaxScaler()
    dis_scaler = MinMaxScaler()
    od_scaler = MinMaxScaler()

    nfeat_scaler = nfeat_scaler.fit(np.concatenate(nfeats, axis=0))
    dis_scaler = dis_scaler.fit(np.concatenate([dis.reshape([-1, 1]) for dis in dises], axis=0))
    od_scaler = od_scaler.fit(np.concatenate([od.reshape([-1, 1]) for od in ods], axis=0))

    return nfeat_scaler, dis_scaler, od_scaler

def build_graph(adjm):
    dst, src = adjm.nonzero()
    d = adjm[adjm.nonzero()]
    g = dgl.graph(([], []))
    g.add_nodes(adjm.shape[0])
    g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1)})
    return g