import glob
import os
import functools

import torch
import torch.nn as nn

from a2c_ppo_acktr.envs import VecNormalize


@functools.lru_cache(maxsize=1)
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compute_feature_size(input_shape, convs):
    return convs(torch.zeros(1, *input_shape)).view(1, -1).size(1)


def get_ball_position(ram):
    ball_x = ram[49]
    ball_y = ram[54]

    return (ball_x, ball_y)


def get_player_paddle_y(ram):
    return ram[51]


def get_cpu_paddle_y(ram):
    return ram[21]


def get_player_score(ram):
    return ram[14]


def get_cpu_score(ram):
    return ram[13]


def get_ball_direction(prev_ram, cur_ram):
    px, py = get_ball_position(prev_ram)
    x, y = get_ball_position(cur_ram)

    if px < x:
        return 1
    return 0


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)
