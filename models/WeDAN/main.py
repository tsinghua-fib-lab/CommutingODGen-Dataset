from dgl.dataloading import GraphDataLoader

from train import *
from eval import test
from utils.procedure import *
from utils.tool import collate_fn
from data_load import *
from model import *
from utils.MyLogger import Logger



config = {
    "exp_path" : "exp/",
    "data_path" : "data/",
    "shuffle_cities" : 0,

    "attr_MinMax" : 1,
    "od_MinMax" : 1,
    "skew_norm" : "log",

    "pert_node" : 1,

    "sample_method" : "DDIM",

    "valid_period" : 1,
    "overfit_tolerance" : 10,
    
    "hiddim" : 32,
    "num_head" : 4,
    "num_head_cross" : 1,
    "num_layer" : 4,
    "dropout" : 0,

    "p_generation" : 1,                                                                                                                                                                                                                                                                        
    "p_featMissing" : 0,


    "train_set": 0.8,
    "valid_set": 0.1,
    "test_set": 0.1,

    "T": 1000,
    "DDIM_T_sample" : 100,
    "sample_times" : 10,
    "DDIM_eta" : 0, 
    "beta_scheduler" : "cosine",

    "EPOCH" : 20000,
    "max_nodes" : 250,
    "city_type_limit": "~250",
    "LaPE_dim" : 0,
    "batch_size" : 32,
    "norm_type" : "layer",
    "learning_rate": 1e-3,
    "optm" : "AdamW",
    "loss" : "mse",
}

# load data
data, trainSet, validSet, testSet, scalers = prepare_data(config)
# validSet = prepare_big(config, scalers) # check small to big
config["num_train_samples"] = len(trainSet)
config["num_valid_samples"] = len(validSet)
config["num_test_samples"] = len(testSet)
# set input dim for model
config["n_indim"] = data[0]["nfeat"].shape[1]
config["e_indim"] = 2
config["n_outdim"] = config["n_indim"]
config["e_outdim"] = 1

# logger
print("  ** preparing logger...", end="")
logger = Logger(config)
print("done")

print("  ** constructing dataloader...", end="")
train_set = MyDataset(trainSet, config, "train")
valid_set = MyDataset(validSet, config, "valid")
test_set = MyDataset(testSet, config, "test")
tnBS = MyBatchSampler(train_set, config["batch_size"], config["max_nodes"])
vdBS = MyBatchSampler(valid_set, config["batch_size"], config["max_nodes"])
ttBS = MyBatchSampler(test_set, config["batch_size"], config["max_nodes"])
train_Dloader = GraphDataLoader(train_set, batch_sampler=tnBS, collate_fn=collate_fn)
valid_Dloader = GraphDataLoader(valid_set, batch_sampler=vdBS, collate_fn=collate_fn)
test_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)
print("done")

# model
print("  ** preparing model...", end="")
diff_model = Diffusion(config).to(torch.device("cuda"))
print("done")

train(config, diff_model, train_Dloader, valid_Dloader, logger, scalers)

test(config, diff_model, test_Dloader, logger, scalers)

print("  ** Finished!")