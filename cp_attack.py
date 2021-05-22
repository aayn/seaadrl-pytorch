import os
import functools
import itertools
from typing import Callable, Iterable
import pickle

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
from evaluation import evaluate


class AttackedAgent:
    def __init__(self, trained_agent):
        self.trained_agent = trained_agent

    def __getattr__(self, attr):
        return getattr(self.trained_agent, attr, None)


class GeneratorDataset(IterableDataset):
    """Uses a generator function to generate batches of data."""

    def __init__(self, generator_fn: Callable):
        self.generator_fn = generator_fn

    def __iter__(self) -> Iterable:
        return self.generator_fn()


def experience_gen(env):
    _ = env.reset()
    prev_obs = env.get_attr("unwrapped")[0]._get_ram()

    action = torch.randint(0, 6, (1,)).unsqueeze(-1)
    while True:
        _, reward, done, _ = env.step(action)
        cur_obs = env.get_attr("unwrapped")[0]._get_ram()
        if done:
            yield prev_obs, action, cur_obs
            _ = env.reset()
            prev_obs = env.get_attr("unwrapped")[0]._get_ram()
        else:
            yield prev_obs, action, cur_obs
            prev_obs = cur_obs


class PredictorMLP(nn.Module):
    def __init__(self, in_size=128, hidden_size=256, out_size=128):
        super().__init__()
        self.fc_grp1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.action_fc = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU())

        self.fc_grp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, obs, action):
        action = action.squeeze(-1)
        obs_encoding = self.fc_grp1(obs)
        action_encoding = self.action_fc(action)
        prod = obs_encoding * action_encoding
        out = self.fc_grp2(prod)
        return out


def RAMLoss(pred, target):
    loss = nn.MSELoss()
    mse = loss(pred, target)

    cpu_score_loss = (pred[:, 13] - target[:, 13]).pow(2).mean()
    player_score_loss = (pred[:, 14] - target[:, 14]).pow(2).mean()
    cpu_paddle_loss = (pred[:, 21] - target[:, 21]).pow(2).mean()
    player_paddle_loss = (pred[:, 51] - target[:, 51]).pow(2).mean()
    ball_pos_loss = (pred[:, 49] - target[:, 49]).pow(2).mean() + (
        pred[:, 54] - target[:, 54]
    ).pow(2).mean()

    return (
        mse
        + 0.1 * cpu_score_loss
        + 0.1 * player_score_loss
        + 0.5 * cpu_paddle_loss
        + 2.0 * player_paddle_loss
        + 2.0 * ball_pos_loss
    )


class CPAttack:
    def __init__(self, config: Box):
        self.conf = config
        device = utils.get_device()
        self.trained_agent, _ = torch.load(
            os.path.join(config.load_dir, config.env_name + ".pt"), map_location=device
        )
        self.trained_agent.eval()
        # self.attacked_agent = AttackedAgent(self.trained_agent)

        # evaluate(self.trained_agent, obs_rms, config.env_name, config.seed, 10, "logs/pong", device)

        # self.env = make_vec_envs(
        #     config.env_name,
        #     config.seed + 1000,
        #     1,
        #     None,
        #     None,
        #     device="cpu",
        #     allow_early_resets=True,
        # )

        # vec_norm = get_vec_normalize(self.env)
        # if vec_norm is not None:
        #     vec_norm.eval()
        #     vec_norm.obs_rms = obs_rms

        # Whether or not we attacked during this life cycle
        self.m = config.M
        self.n = config.N

    def run(self, rseed=1, threshold=5.0):
        prev_player_score = 0
        prev_cpu_score = 0
        can_attack = True

        env = make_vec_envs(
            self.conf.env_name,
            self.conf.seed + rseed,
            1,
            None,
            None,
            device=utils.get_device(),
            allow_early_resets=False,
        )

        done = False
        obs = env.reset()

        attack_counts = 0
        while not done:
            actions, done, can_attack, did_attack = self.get_next_m_actions(
                env, obs, can_attack, threshold
            )
            if did_attack:
                attack_counts += 1
            for action in actions:
                next_obs, reward, _, _ = env.step(action)
                obs = next_obs
                if "terminal_observation" in env.buf_infos[0]:
                    done = True
                    break
            ram = env.get_attr("unwrapped")[0]._get_ram()
            cpu_score = utils.get_cpu_score(ram)
            player_score = utils.get_player_score(ram)

            if cpu_score != prev_cpu_score or player_score != prev_player_score:
                print(cpu_score, player_score, attack_counts)
                if not done:
                    prev_cpu_score, prev_player_score = cpu_score, player_score
                can_attack = True

        return int(prev_cpu_score), int(prev_player_score), attack_counts

    def get_next_m_actions(self, env, init_obs, can_attack=True, threshold=5.0):
        clone_fns = env.get_attr("clone_full_state")
        init_env_states = [cf() for cf in clone_fns]
        baseline_actions = []

        obs = init_obs
        for i in range(self.m):
            with torch.no_grad():
                _, action, _, _ = self.trained_agent.act(
                    obs, None, None, deterministic=True
                )
            next_obs, reward, dones, _ = env.step(action)
            baseline_actions.append(action)
            done = dones[0]
            if i == 0 and done:
                break
            obs = next_obs

        if not can_attack or (i == 0 and done):
            return baseline_actions, done, can_attack, False

        expected_state = env.get_attr("unwrapped")[0]._get_ram()
        expected_divergence = self.calc_divergence(expected_state)

        for act_seq in self.all_action_seqs(env.action_space.n):
            restore_fns = env.get_attr("restore_full_state")
            for rs, rf in zip(init_env_states, restore_fns):
                rf(rs)
            obs = init_obs
            attack_actions = []

            for action in act_seq:
                action = torch.tensor([[action]])
                next_obs, reward, dones, _ = env.step(action)
                attack_actions.append(action)
                done = dones[0]
                obs = next_obs

            attacked_state = env.get_attr("unwrapped")[0]._get_ram()
            attacked_divergence = self.calc_divergence(attacked_state)

            delta = abs(expected_divergence - attacked_divergence)
            # if delta > 0:
            #     print(
            #         f"Baseline: {baseline_actions}; Attack: {act_seq}; Delta = {delta}"
            #     )

            if can_attack and delta > threshold:
                restore_fns = env.get_attr("restore_full_state")
                for rs, rf in zip(init_env_states, restore_fns):
                    rf(rs)
                return attack_actions, done, False, True

        restore_fns = env.get_attr("restore_full_state")
        for rs, rf in zip(init_env_states, restore_fns):
            rf(rs)

        return baseline_actions, done, can_attack, False

    def all_action_seqs(self, num_actions):
        yield from itertools.product(range(num_actions), repeat=self.m)

    def calc_divergence(self, ram):
        bx, by = utils.get_ball_position(ram)
        px, py = 190, utils.get_cpu_paddle_y(ram) + 12

        # prob = 1.0 if bx >= 192 else 0.0
        prob = min(np.exp(bx - 192), 1.0)
        dist = np.sqrt(pow((bx - px), 2) + pow((by - py), 2))

        return prob * dist

    def train_predictor(self):
        torch.manual_seed(self.conf.seed + 7)
        torch.cuda.manual_seed_all(self.conf.seed + 7)
        device = utils.get_device()
        # torch.set_num_threads(1)

        envs = make_vec_envs(
            self.conf.env_name,
            self.conf.seed + 7,
            1,
            0.99,
            "logs/pong/train_cp_predictor",
            device,
            False,
        )

        dataset = GeneratorDataset(functools.partial(experience_gen, env=envs))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        net = PredictorMLP().to(device)
        lr = 1e-5
        optimizer = optim.Adam(net.parameters(), lr=lr)

        for i, (prev, action, cur) in zip(range(200000), dataloader):
            prev = prev.float().to(device)
            action = action.float().to(device)
            cur = cur.float().to(device)
            pred = net(prev, action)

            optimizer.zero_grad()
            out = RAMLoss(pred, cur)
            out.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"{i}, {out.item()}")


def main():
    with open("seaadrl.yaml") as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader)["cp-attack"])

    cp = CPAttack(config)
    # cp.train_predictor()

    cp_log = {}

    # cpu_score, player_score = cp.run(0, 0)

    for threshold in itertools.chain(np.arange(0, 2, 0.25), range(2, 21)):
        scores = []
        for rseed in range(3):
            cpu_score, player_score, attack_counts = cp.run(rseed, threshold)
            scores.append((player_score - cpu_score, attack_counts))
            print(threshold, scores)
        cp_log[threshold] = scores

    with open("logs/pong/plots/cp_test.pkl", "wb") as pf:
        pickle.dump(cp_log, pf)


if __name__ == "__main__":
    main()