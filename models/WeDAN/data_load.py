import os
from random import shuffle, choice

import numpy as np

import torch
import dgl

from tqdm import tqdm

from utils.tool import *


def prepare_data(config):

    if config["skew_norm"] == "none":
        OD_normer = None_transformer()
    elif config["skew_norm"] == "log":
        OD_normer = Log_transformer()
    elif config["skew_norm"] == "boxcox":
        OD_normer = BoxCox_transformer()
    else:
        raise Exception("Unknown skew_norm type")


    cities = os.listdir(config["data_path"])
    if config["shuffle_cities"] == 1:
        shuffle(cities)

    # load data
    print("***************** load data *****************")
    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "nfeat": np.concatenate((np.load(config["data_path"] + city + "/demos.npy"), np.load(config["data_path"] + city + "/pois.npy")), axis=1),
                "dis": np.load(config["data_path"] + city + "/dis.npy"),
                "od": OD_normer.fit_transform(np.load(config["data_path"] + city + "/od.npy"))
            }
        
        for i in range(one["od"].shape[0]):
            one["od"][i,i] = 0

        data.append(one)

    print("  ** constructing dataset...", end="")
    train, valid, test = split_data_intoTVT(data, config)
    print("done")

    # normalization
    print("  ** normalizing dataset...", end="")
    if config["attr_MinMax"] == 1:
        scaler_feat = MinMaxer(np.concatenate([x["nfeat"] for x in train], axis=0))
        for i, nfeat in enumerate([x["nfeat"] for x in train]):
            train[i]["nfeat"] = scaler_feat.transform(nfeat)
        for i, nfeat in enumerate([x["nfeat"] for x in valid]):
            valid[i]["nfeat"] = scaler_feat.transform(nfeat)
        for i, nfeat in enumerate([x["nfeat"] for x in test]):
            test[i]["nfeat"] = scaler_feat.transform(nfeat)
        scaler_dis = MinMaxer(np.concatenate([x["dis"].reshape([-1, 1]) for x in train], axis=0))
        for i, dis in enumerate([x["dis"] for x in train]):
            train[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
        for i, dis in enumerate([x["dis"] for x in valid]):
            valid[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
        for i, dis in enumerate([x["dis"] for x in test]):
            test[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    else:
        scaler_feat = None
        scaler_dis = None

    if config["od_MinMax"] == 1:    
        scaler_od = MinMaxer(np.concatenate([x["od"].reshape([-1, 1]) for x in train], axis=0))
        for i, od in enumerate([x["od"] for x in train]):
            train[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        for i, od in enumerate([x["od"] for x in valid]):
            valid[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        for i, od in enumerate([x["od"] for x in test]):
            test[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
    else:
        scaler_od = None
    scalers = {
        "feat" : scaler_feat,
        "dis" : scaler_dis,
        "od" : scaler_od,
        "od_normer" : OD_normer,
    }
    print("done")
    
    return data, train, valid, test, scalers


class MyDataset(dgl.data.DGLDataset):
    def __init__(self, data, config, Type):
        self.config = config
        self.data = data
        self.type = Type
        super(MyDataset,self).__init__(name="OD_graph_dataset")

    def process(self):
        self.GEOIDs = []
        self.graphs = []
        self.dises = []
        self.ods = []
        for one in self.data:
            self.GEOIDs.append(one["GEOID"])
            g = dgl.graph(one["od"].nonzero(), num_nodes=one["od"].shape[0])
            g.ndata["demo"] = torch.FloatTensor(one["nfeat"])
            self.graphs.append(g)
            self.dises.append(torch.FloatTensor(one["dis"]))
            self.ods.append(torch.FloatTensor(one["od"]))

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if self.type != "train":
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]
        if self.config["pert_node"] == 1:
            geoid = self.GEOIDs[index]
            graph = self.graphs[index]
            dis = self.dises[index]
            od = self.ods[index]
            graph, dis, od = permut_nodes_order(graph, dis, od)
            return geoid, graph, dis, od
        else:
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]


class MyBatchSampler:
    def __init__(self, dataset, batch_size, max_value):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_value = max_value

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                yield batch

        if batch:
            yield batch
    
    def __len__(self):
        indices = list(range(len(self.dataset)))
        count = 0

        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                count += 1

        return count