import os
import json
from copy import deepcopy

import numpy as np

import geopandas as gpd

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import accuracy
from utils.tool import *


class Logger():
    def __init__(self, config):
        self.config = config

        # records
        self.training_losses = []
        self.valid_losses = []
        self.valid_metrics = []
        self.best_valid_gen_cpc = 0
        self.best_valid_comp_cpc = 0
        self.best_valid_missgen_cpc = 0
        self.best_valid_misscomp_cpc = 0
        self.pre_train_losses = []
        self.pre_train_metrics = {}

        # early stopping
        self.overfit_flag = 0
        
        # log dicts
        self.train_log = {
            "num_epochs" : 0,

            "train_loss" : self.training_losses,
            "valid_loss" : self.valid_losses,

            "best_valid_gen_cpc" : self.best_valid_gen_cpc,
            "best_valid_comp_cpc" : self.best_valid_comp_cpc,
            "best_valid_missgen_cpc" : self.best_valid_missgen_cpc,
            "best_valid_misscomp_cpc" : self.best_valid_misscomp_cpc,

            "valid_metrics" : self.valid_metrics
        }

        self.exp_log = {
                         "config" : self.config,
                         "train_log" : self.train_log
        }


    # valid log
    def once_valid_record(self, epoch, loss, metrics, model):
        # log
        self.valid_losses.append({epoch : float(loss)})
        self.valid_metrics.append({epoch : metrics})

        self.overfit_flag = self.overfit_flag + 1
        # saving model optm
        if metrics["gen"]["all"]["CPC"] > self.best_valid_gen_cpc:
            self.best_valid_gen_cpc = metrics["gen"]["all"]["CPC"]
            self.best_model = deepcopy(model)
            self.overfit_flag = 0