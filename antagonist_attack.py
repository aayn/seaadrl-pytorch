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
from evaluation import evaluate, evaluate_antagonist


class AntagonistAttack:
    def __init__(self, config: Box):
        self.conf = config
        device = utils.get_device()
        self.trained_agent, _ = torch.load(
            os.path.join(config.load_dir, config.env_name + ".pt"), map_location=device
        )
        self.trained_agent.eval()

        # Whether or not we attacked during this life cycle
        self.n = config.N

    def train_defence(self):
        envs = make_vec_envs(
            self.conf.env_name,
            self.conf.seed,
            self.conf.train_num_envs,
            self.conf.gamma,
            "/tmp/gym",
            device=utils.get_device(),
            allow_early_resets=False,
            antagonist=True,
        )

        ac_vaxx = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={"recurrent": False},
        )
        ac_vaxx.to(utils.get_device())

        agent_vaxx = algo.A2C_ACKTR(
            ac_vaxx,
            self.conf.value_loss_coef,
            self.conf.entropy_coef,
            lr=self.conf.lr,
            eps=self.conf.eps,
            alpha=self.conf.alpha,
            max_grad_norm=self.conf.max_grad_norm,
        )

        antagonist, _, _ = torch.load(
            os.path.join(
                self.conf.load_dir,
                self.conf.env_name + f"-det-bugfix-antagonist_{self.n}.pt",
            ),
            map_location=utils.get_device(),
        )
        antagonist.eval()

        attacks_left = (
            torch.tensor([self.n] * envs.num_envs, dtype=torch.float)
            .reshape(envs.num_envs, 1)
            .to(utils.get_device())
        )

        antagonist.to(utils.get_device())
        anti_agent = algo.A2C_ACKTR(
            antagonist,
            self.conf.value_loss_coef,
            self.conf.entropy_coef,
            lr=self.conf.lr,
            eps=self.conf.eps,
            alpha=self.conf.alpha,
            max_grad_norm=self.conf.max_grad_norm,
        )

        rollouts = RolloutStorage(
            self.conf.num_steps,
            envs.num_envs,
            envs.observation_space.shape,
            envs.action_space,
            antagonist.recurrent_hidden_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(utils.get_device())

        episode_rewards = deque(maxlen=10)
        total_attacks = 0

        start = time.time()
        num_updates = (
            int(self.conf.total_steps)
            // self.conf.num_steps
            // self.conf.train_num_envs
        )

        num_uniq = 1
        attack_threshold = 0.5

        for j in range(num_updates):
            if self.conf.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    anti_agent.optimizer,
                    j,
                    num_updates,
                    utils.get_lr(anti_agent.optimizer),
                )
            for step in range(self.conf.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        ant_value,
                        ant_action,
                        ant_action_log_prob,
                        ant_recurrent_hidden_states,
                    ) = antagonist.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

                    (
                        tr_value,
                        tr_action,
                        tr_action_log_prob,
                        tr_recurrent_hidden_states,
                    ) = ac_vaxx.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        deterministic=True,
                    )

                    ant_action_mask = (
                        ant_action_log_prob > np.log(attack_threshold)
                    ) & (attacks_left > 0)

                    # print(ant_action_mask)
                    # print(max(torch.exp(ant_action_log_prob)))

                    total_attacks += sum(ant_action_mask).item()

                    attacks_left -= (
                        torch.ones(envs.num_envs, 1).to(utils.get_device())
                        * ant_action_mask
                    )

                    # value = ant_action_mask * ant_value + ~ant_action_mask * value
                    # value = ant_value

                    action = ant_action_mask * ant_action + ~ant_action_mask * tr_action

                    # value, action_log_prob, _, _ = antagonist.evaluate_actions(rollouts.obs[step], None, None, ant_action)

                    # action_log_prob = (
                    #     ant_action_mask * ant_action_log_prob
                    #     + ~ant_action_mask * action_log_prob
                    # )

                    # action_log_prob = ant_action_log_prob

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                # reward *= -1

                num_uniq = len(torch.unique(action))

                # print(reward)

                # reset num attacks
                scored_mask = reward != 0

                attacks_left[scored_mask] = self.n

                for info in infos:
                    if "episode" in info.keys():
                        episode_rewards.append(info["episode"]["r"])

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if "bad_transition" in info.keys() else [1.0]
                        for info in infos
                    ]
                )
                rollouts.insert(
                    obs,
                    tr_recurrent_hidden_states,
                    tr_action,
                    tr_action_log_prob,
                    tr_value,
                    reward,
                    masks,
                    bad_masks,
                )

            with torch.no_grad():
                next_value = ac_vaxx.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value,
                False,
                self.conf.gamma,
                0.95,
                False,
            )

            value_loss, action_loss, dist_entropy = agent_vaxx.update(rollouts)
            rollouts.after_update()
            if (
                j % self.conf.save_interval == 0 or j == num_updates - 1
            ) and self.conf.save_dir != "":
                save_path = os.path.join(self.conf.save_dir, self.conf.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(
                    [
                        ac_vaxx,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(
                        save_path,
                        self.conf.env_name + f"-vaxxed_{self.n}.pt",
                    ),
                )

            if j % self.conf.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (
                    (j + 1) * self.conf.train_num_envs * self.conf.num_steps
                )
                end = time.time()
                lr = utils.get_lr(anti_agent.optimizer)
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy: {:.3f}, critic loss: {:.3f}, actor loss: {:.3f}, total attacks: {}, lr: {}, unique_actions: {}\n".format(
                        j,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        dist_entropy,
                        value_loss,
                        action_loss,
                        total_attacks,
                        lr,
                        num_uniq,
                    )
                )

            # if (
            #     self.conf.eval_interval is not None
            #     and len(episode_rewards) > 1
            #     and j % self.conf.eval_interval == 0
            # ):
            #     obs_rms = utils.get_vec_normalize(envs).obs_rms
            #     evaluate(
            #         actor_critic,
            #         obs_rms,
            #         args.env_name,
            #         args.seed,
            #         args.num_processes,
            #         eval_log_dir,
            #         device,
            #     )

    def evaluate(self):
        antagonist, attack_threshold, _ = torch.load(
            os.path.join(
                self.conf.load_dir,
                self.conf.env_name + f"-det-bugfix-antagonist_{self.n}.pt",
            ),
            map_location=utils.get_device(),
        )
        antagonist.eval()
        attack_threshold = 0.5
        evaluate_antagonist(
            self.trained_agent,
            antagonist,
            self.n,
            attack_threshold,
            None,
            self.conf.env_name,
            self.conf.seed,
            self.conf.eval_num_envs,
            "/tmp/gym",
            utils.get_device(),
        )

    def train(self):
        envs = make_vec_envs(
            self.conf.env_name,
            self.conf.seed,
            self.conf.train_num_envs,
            self.conf.gamma,
            "/tmp/gym",
            device=utils.get_device(),
            allow_early_resets=False,
            antagonist=True,
        )

        attacks_left = (
            torch.tensor([self.n] * envs.num_envs, dtype=torch.float)
            .reshape(envs.num_envs, 1)
            .to(utils.get_device())
        )

        antagonist = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={"recurrent": False},
        )
        antagonist.to(utils.get_device())
        anti_agent = algo.A2C_ACKTR(
            antagonist,
            self.conf.value_loss_coef,
            self.conf.entropy_coef,
            lr=self.conf.lr,
            eps=self.conf.eps,
            alpha=self.conf.alpha,
            max_grad_norm=self.conf.max_grad_norm,
        )

        rollouts = RolloutStorage(
            self.conf.num_steps,
            envs.num_envs,
            envs.observation_space.shape,
            envs.action_space,
            antagonist.recurrent_hidden_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(utils.get_device())

        episode_rewards = deque(maxlen=10)
        total_attacks = 0

        start = time.time()
        num_updates = (
            int(self.conf.total_steps)
            // self.conf.num_steps
            // self.conf.train_num_envs
        )

        num_uniq = 1
        attack_threshold = 0.197

        for j in range(num_updates):
            if self.conf.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    anti_agent.optimizer,
                    j,
                    num_updates,
                    utils.get_lr(anti_agent.optimizer),
                )
            for step in range(self.conf.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        ant_value,
                        ant_action,
                        ant_action_log_prob,
                        ant_recurrent_hidden_states,
                    ) = antagonist.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

                    (
                        tr_value,
                        tr_action,
                        tr_action_log_prob,
                        tr_recurrent_hidden_states,
                    ) = self.trained_agent.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        deterministic=True,
                    )

                    ant_action_mask = (
                        ant_action_log_prob > np.log(attack_threshold)
                    ) & (attacks_left > 0)

                    # print(ant_action_mask)
                    # print(max(torch.exp(ant_action_log_prob)))

                    total_attacks += sum(ant_action_mask).item()

                    attacks_left -= (
                        torch.ones(envs.num_envs, 1).to(utils.get_device())
                        * ant_action_mask
                    )

                    # value = ant_action_mask * ant_value + ~ant_action_mask * value
                    # value = ant_value

                    action = ant_action_mask * ant_action + ~ant_action_mask * tr_action

                    # value, action_log_prob, _, _ = antagonist.evaluate_actions(rollouts.obs[step], None, None, ant_action)

                    # action_log_prob = (
                    #     ant_action_mask * ant_action_log_prob
                    #     + ~ant_action_mask * action_log_prob
                    # )

                    # action_log_prob = ant_action_log_prob

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                reward *= -1

                num_uniq = len(torch.unique(action))

                # print(reward)

                # reset num attacks
                scored_mask = reward != 0

                attacks_left[scored_mask] = self.n

                for info in infos:
                    if "episode" in info.keys():
                        episode_rewards.append(info["episode"]["r"])

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if "bad_transition" in info.keys() else [1.0]
                        for info in infos
                    ]
                )
                rollouts.insert(
                    obs,
                    ant_recurrent_hidden_states,
                    ant_action,
                    ant_action_log_prob,
                    ant_value,
                    reward,
                    masks,
                    bad_masks,
                )

            with torch.no_grad():
                next_value = antagonist.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value,
                False,
                self.conf.gamma,
                0.95,
                False,
            )

            value_loss, action_loss, dist_entropy = anti_agent.update(rollouts)
            rollouts.after_update()
            if (
                j % self.conf.save_interval == 0 or j == num_updates - 1
            ) and self.conf.save_dir != "":
                save_path = os.path.join(self.conf.save_dir, self.conf.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(
                    [
                        antagonist,
                        attack_threshold,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(
                        save_path,
                        self.conf.env_name + f"-det-bugfix-antagonist_{self.n}.pt",
                    ),
                )

            if j % self.conf.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (
                    (j + 1) * self.conf.train_num_envs * self.conf.num_steps
                )
                end = time.time()
                lr = utils.get_lr(anti_agent.optimizer)
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy: {:.3f}, critic loss: {:.3f}, actor loss: {:.3f}, total attacks: {}, lr: {}, unique_actions: {}\n".format(
                        j,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        dist_entropy,
                        value_loss,
                        action_loss,
                        total_attacks,
                        lr,
                        num_uniq,
                    )
                )

            # if (
            #     self.conf.eval_interval is not None
            #     and len(episode_rewards) > 1
            #     and j % self.conf.eval_interval == 0
            # ):
            #     obs_rms = utils.get_vec_normalize(envs).obs_rms
            #     evaluate(
            #         actor_critic,
            #         obs_rms,
            #         args.env_name,
            #         args.seed,
            #         args.num_processes,
            #         eval_log_dir,
            #         device,
            #     )


def main():
    torch.set_num_threads(1)

    with open("seaadrl.yaml") as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader)["antagonist-attack"])

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # for n in range(3, 0, -1):
    #     config.N = n
    # ant = AntagonistAttack(config)
    # ant.train()
    ant = AntagonistAttack(config)
    # ant.evaluate()
    ant.train_defence()


if __name__ == "__main__":
    main()