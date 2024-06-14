import os
import sys
import json

import random
import argparse

import numpy as np
import torch


def getoneGPUMem():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    total_oneGPUMem = int(gpu_status[2].replace(" ", "").split("/")[1][:-3])
    return total_oneGPUMem


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
        return []


def choose_gpu_from(candis = None, mem_need = 10000):
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
    if candis:
        id_mem = id_mem[candis, :]
    GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]
    
    i = 0
    while len(GPU_available) < 1:  # set waiting condition

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
        if candis:
            id_mem = id_mem[candis, :]
        GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]

        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        i += 1
    GPU_selected = random.choice(list(GPU_available))

    return GPU_selected


def narrow_setup(mem_need = 10000):
    GPU_available = gpu_info(mem_need)
    i = 0
    while len(GPU_available) < 1:  # set waiting condition
        GPU_available = gpu_info(mem_need)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        i += 1
    GPU_selected = random.choice(GPU_available)

    return GPU_selected

def get_config(path):
    config = json.load(open(path, "r"))

    # check GPU available
    if config["check_device"] == 1:
        GPU_no = narrow_setup(mem_need=config["mem_need"])
        config["device"] = torch.device(int(GPU_no))
    else:
        config["device"] = torch.device(int(config["device"]))

    return config
    
def setRandomSeed(seed):
    # one random_seed in config file
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("****** set random seed as", seed, " ******\n")
    return seed


def add_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=1, help='Graph transformer layers.')
    parser.add_argument('--hiddim', type=int, default=8, help='Graph transformer layers.')
    parser.add_argument('--cities', type=int, default=10, help='Graph transformer layers.')
    args = parser.parse_args()

    return args