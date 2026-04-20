import numpy as np
import torch
import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def log_experiment(config):
    print("Experiment configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
