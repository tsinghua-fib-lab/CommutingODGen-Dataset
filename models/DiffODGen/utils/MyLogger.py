import os
import json
from copy import deepcopy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import accuracy


class Logger():
    def __init__(self, config):
        self.config = config

        self.topo_epochs = 0
        self.flow_epochs = 0
        self.training_losses_topo = []
        self.training_losses_flow = []
        self.best_training_loss = float("inf")

        self.valid_errors = []
        self.best_valid_error = float("inf")

        self.best_test_cpc_topo = 0
        self.best_test_cpc_flow = 0
      
        self.whether_best_flags = []

        self.whether_overfitting_flags = []

        self.train_log = {
            "topo_epochs" : self.topo_epochs,
            "flow_epochs" : self.flow_epochs,
            "train_loss_topo" : self.training_losses_topo,
            "train_loss_flow" : self.training_losses_flow,
            
            "valid_errors" : self.valid_errors,
            "best_valid_error" : self.best_valid_error,
            "whether_best_flags" : self.whether_best_flags,
            "whether_overfitting_flags" : self.whether_overfitting_flags
        }

        self.exp_log = {
                         "config" : self.config,
                         "train_log" : self.train_log,
                         "eval_log": {
                            "RMSE" : float("inf"),
                            "NRMSE" : float("inf"),
                            "MAE" : float("inf"),
                            "MAPE" : float("inf"),
                            "SMAPE" : float("inf"),
                            "CPC" : 0, 
                            "RMSE_nonzero" : float("inf")
                         }
        }

    def log_training_loss(self, current_loss, exp):
        if exp == "topo":
            self.training_losses_topo.append(float(current_loss))
        elif exp == "flow":
            self.training_losses_flow.append(float(current_loss))

    def log_valid_errors(self, rmse):
        print("valid_errors = ", rmse)
        self.valid_errors.append(float(rmse))

    def log_results_topo(self, metrics):
        for k, v in metrics.items():
            self.exp_log["eval_log"][k] = float(v)

    def log_results_flow(self, metrics):
        for k, v in metrics.items():
            self.exp_log["eval_log"][k] = float(v)

    def check_save_model(self, current_loss, model, optimizer):

        if current_loss < self.best_training_loss:
            self.best_training_loss = current_loss
            self.whether_best_flags = []
            torch.save(model.state_dict(), self.model_path)
            torch.save(optimizer.state_dict(), self.optimizer_path)
            print("Best loss ever and save the model.")
        else:
            self.whether_best_flags.append(1)
    
    def save_model_optm_scheduler(self, model, optimizer, exp):
        torch.save(model.state_dict(), self.model_path[:-4] + "_" + exp + ".pkl")
        torch.save(optimizer.state_dict(), self.optimizer_path[:-4] + "_" + exp + ".pkl")

    def check_overfitting(self, current_valid_errors):
        self.log_valid_errors(current_valid_errors)
        if current_valid_errors < self.best_valid_error:
            self.best_valid_error = current_valid_errors
            self.whether_overfitting_flags = []
        else:
            self.whether_overfitting_flags.append(1)

        if len(self.whether_overfitting_flags) > 500:
            print("Overfitting!")
            return True
        else:
            print("Overfitting : " + int(len(self.whether_overfitting_flags) * (2 / 5)) * "@" + (20 - int(len(self.whether_overfitting_flags) * (2 / 5))) * "-")
            return False

    def check_converge(self):
        if len(self.whether_best_flags) > self.config["converge_check"]:
            print("Converged!!!")
            return True
        else:
            already = int(len(self.whether_best_flags) / self.config["converge_check"] * 20)
            print("Convergence : " + already * "@" + (20 - already) * "-")
            return False

    def clear_check(self):
        self.whether_overfitting_flags = []
        self.whether_best_flags = []

    def save_exp_log(self):
        exp_log = deepcopy(self.exp_log)
        
        if "device" in exp_log["config"].keys():
            exp_log["config"]["device"] = int(exp_log["config"]["device"].index)
            if "devices" in exp_log["config"].keys():
                exp_log["config"].pop("devices")
        else:
            exp_log["config"]["devices"] = [int(x.index) for x in exp_log["config"]["devices"]]
        json.dump(exp_log, open(self.exp_path, "w"), indent=4)
    
    def load_exp_log(self):
        print("*********** load exp log **************", "\n")
        self.exp_log = json.load(open(self.exp_path, "r"))
        if "device" in self.exp_log["config"].keys():
            self.exp_log["config"]["device"] = torch.device(self.exp_log["config"]["device"])
        else:
            self.exp_log["config"]["devices"] = [torch.device(x) for x in self.exp_log["config"]["devices"]]
        self.training_losses_topo = self.exp_log["train_log"]["train_loss_topo"]
        self.training_losses_flow = self.exp_log["train_log"]["train_loss_flow"]

    def load_exp_tensorboard(self):
        print("*********** load exp TensorBoard **************")
        train_log = self.exp_log["train_log"]
        topo_losses = train_log["train_loss_topo"]
        flow_losses = train_log["train_loss_flow"]
        for i, loss in enumerate(topo_losses):
            self.summary_record(loss, name="topo_Train/loss", iteration=i)
        for i, loss in enumerate(flow_losses):
            self.summary_record(loss, name="flow_Train/loss", iteration=i)
        print()

    def summary_record(self, variable, name, iteration):
        self.summary_writer.add_scalar(name, variable, iteration)

    def summary_all_metrics(self, metrics, epoch):
        for k, v in metrics.items():
            self.summary_record(v, "Valid/"+k, epoch)
    
    def summary_all_test_metrics(self, metrics, epoch, tag):
        for k, v in metrics.items():
            self.summary_record(v, "flow_Test/" + tag + "/" + k, epoch)

    def once_record_flow(self, training_loss, model, optimizer, **kargs):
        self.log_training_loss(current_loss=training_loss, exp="flow")
        self.save_model_optm_scheduler(model, optimizer, exp="flow")
        self.summary_record(training_loss, "flow_Train/loss", len(self.training_losses_flow))

    def once_record_topo(self, training_loss, model, optimizer, **kargs):
        self.log_training_loss(current_loss=training_loss, exp="topo")
        self.save_model_optm_scheduler(model, optimizer, exp="topo")
        self.summary_record(training_loss, "topo_Train/loss", len(self.training_losses_topo))

    def cal_topo_metrics(self, pred, target, epoch, name):
        jac_sim = accuracy(pred, target)
        self.summary_record(jac_sim, name, epoch)
    
    def summary_all_test_metrics_topo(self, metrics, epoch, tag):
        for k, v in metrics.items():
            self.summary_record(v, "topo_Test/" + tag + "/" + k, epoch)

    def upgrade_topo_epochs(self):
        self.topo_epochs += 1
    
    def upgrade_flow_epochs(self):
        self.flow_epochs += 1

    def save_generation(self, city, generation, tag):
        if not os.path.exists(self.generation_directory):
            os.mkdir(self.generation_directory[:-1])
        generation = generation.cpu().numpy()
        np.save(self.generation_directory + city + "_" + tag + ".npy", generation)