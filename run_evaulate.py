import os
import time
from collections import deque
import functools
import itertools
from typing import Callable, Iterable

import numpy as np
import yaml
import gym
from box import Box

import torch

# torch.multiprocessing.set_start_method("forkserver")
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch import optim


from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    with open("seaadrl.yaml") as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader)["baseline"])

    device = utils.get_device()
    # trained_agent, _ = torch.load(
    #     os.path.join(config.load_dir, config.env_name + ".pt"), map_location=device
    # )
    # trained_agent.eval()

    trained_agent, _ = torch.load(
        os.path.join(config.load_dir, config.env_name + "-vaxxed_2.pt"), map_location=device
    )
    trained_agent.eval()

    evaluate(
        trained_agent,
        None,
        config.env_name,
        seed=1,
        num_processes=24,
        eval_log_dir='/tmp/gym',
        device=utils.get_device(),
    )

if __name__ == "__main__":
    main()