import json
import os
import random
import sys
from pprint import pprint

import numpy as np
import torch


def gpu_info(mem_need = 10000):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    total_oneGPUMem = int(gpu_status[2].replace(" ", "").split("/")[1][:-3])

    mem_idx = [2, 6, 10, 14, 18, 22, 26, 30]
    mem_bus = [x for x in range(torch.cuda.device_count())]
    mem_list = []
    for idx, info in enumerate(gpu_status):
        if idx in mem_idx:
            mem_list.append(total_oneGPUMem - int(info.split('/')[0].split('M')[0].strip()))
    idx = np.array(mem_bus).reshape([-1, 1])
    mem = np.array(mem_list).reshape([-1, 1])
    id_mem = np.concatenate((idx, mem), axis=1)
    GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]

    if len(GPU_available) != 0:
        return GPU_available.tolist()
    else:
        return None

def narrow_setup(config, interval = 0.5, mem_need = 10000):
    GPU_available = gpu_info(mem_need)
    i = 0
    while len(GPU_available) < config["device_num"]:  # set waiting condition
        GPU_available = gpu_info(mem_need)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        # time.sleep(interval)
        i += 1
    if config["device_num"] == 1:
        GPU_selected = random.choice(GPU_available)
    else:
        GPU_selected = random.sample(GPU_available, config["device_num"])
    return GPU_selected

def get_config(path):
    config = json.load(open(path, "r"))
    print("\n****** experiment name:", config["exp_name"], " ******")

    # check GPU available
    if config["check_device"] == 1:
        if config["device_num"] == 1:
            GPU_no = narrow_setup(config, interval = 1, mem_need = config["mem_need"])
            config["device"] = torch.device(int(GPU_no))
            print("\n****** Using No.", int(GPU_no), "GPU ******")
        else:
            GPUs = narrow_setup(config, interval = 1, mem_need = config["mem_need"])
            devices = []
            for gpu in GPUs:
                devices.append(torch.device(int(gpu)))
            config["devices"] = devices
            print("\n****** Using", GPUs, "GPU ******")

    if "device" in config.keys():
        devices = []
        for i in range(8):
            devices.append(config["device"])
        config["devices"] = devices

    print("\n", "****** exp config ******")
    pprint(config)
    print("*************************\n")
    return config
    
def setRandomSeed(seed):
    # one random_seed in config file
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("****** set random seed as", seed, " ******\n")
    return seed

