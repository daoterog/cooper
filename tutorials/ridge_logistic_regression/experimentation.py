"""
Experimentation module
"""

import random

import wandb
import torch
import numpy as np

from pipeline import experimentation_pipeline

# Ensure Reproducibility
random.seed(777)
np.random.seed(777)
torch.manual_seed(777)
torch.backends.cudnn.deterministic = True

# Wandb Login
wandb.login()

if __name__ == "__main__":

    config = dict(
        is_constrained=False,
        unconst_lr=1e-3,
        batch_size = 32,
        n_iters = 1000,
        train_ratio = 0.6,
    )

    # Run Experiment
    model = experimentation_pipeline(config)
