import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(
    actor_critic,
    obs_rms,
    env_name,
    seed,
    num_processes,
    eval_log_dir,
    device,
    antagonist=None,
):
    eval_envs = make_vec_envs(
        env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True
    )

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 200:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

        print(eval_episode_rewards)
        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()

    print(
        " Evaluation using {} episodes: mean reward {:.5f}, stddev reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards), np.std(eval_episode_rewards)
        )
    )


def evaluate_antagonist(
    trained_agent,
    antagonist,
    num_attacks,
    attack_threshold,
    obs_rms,
    env_name,
    seed,
    num_processes,
    eval_log_dir,
    device,
):
    eval_envs = make_vec_envs(
        env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True
    )

    trained_agent.eval()
    antagonist.eval()

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, antagonist.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)

    attacks_left = (
        torch.tensor([num_attacks] * num_processes, dtype=torch.float)
        .reshape(num_processes, 1)
        .to(utils.get_device())
    )

    total_actions = 0
    total_attacks = 0

    while len(eval_episode_rewards) < 32:
        with torch.no_grad():
            (
                _,
                ant_action,
                ant_action_log_prob,
                eval_recurrent_hidden_states,
            ) = antagonist.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

            _, tr_action, _, _, = trained_agent.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )
        ant_action_mask = (ant_action_log_prob > np.log(attack_threshold)) & (
            attacks_left > 0
        )

        total_actions += len(ant_action_mask)
        total_attacks += sum(ant_action_mask).item()
        print(total_attacks, total_actions)

        attacks_left -= (
            torch.ones(num_processes, 1).to(utils.get_device()) * ant_action_mask
        )

        action = ant_action_mask * ant_action + ~ant_action_mask * tr_action
        # action = ant_action

        print(eval_episode_rewards)
        # Obser reward and next obs

        obs, reward, done, infos = eval_envs.step(action)

        # reset num attacks
        scored_mask = reward != 0
        attacks_left[scored_mask] = num_attacks

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()

    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)
        )
    )


def evaluate_antagonist_defence(
    trained_agent,
    antagonist,
    num_attacks,
    attack_threshold,
    obs_rms,
    env_name,
    seed,
    num_processes,
    eval_log_dir,
    device,
):
    eval_envs = make_vec_envs(
        env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True
    )

    trained_agent.eval()
    antagonist.eval()

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, antagonist.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.zeros(num_processes, 1, device=device)

    attacks_left = (
        torch.tensor([num_attacks] * num_processes, dtype=torch.float)
        .reshape(num_processes, 1)
        .to(utils.get_device())
    )

    total_actions = 0
    total_attacks = 0

    while len(eval_episode_rewards) < 32:
        with torch.no_grad():
            (
                _,
                ant_action,
                ant_action_log_prob,
                eval_recurrent_hidden_states,
            ) = antagonist.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

            _, tr_action, _, _, = trained_agent.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )
        ant_action_mask = (ant_action_log_prob > np.log(attack_threshold)) & (
            attacks_left > 0
        )

        total_actions += len(ant_action_mask)
        total_attacks += sum(ant_action_mask).item()
        print(total_attacks, total_actions)

        attacks_left -= (
            torch.ones(num_processes, 1).to(utils.get_device()) * ant_action_mask
        )

        action = ant_action_mask * ant_action + ~ant_action_mask * tr_action
        # action = ant_action

        print(eval_episode_rewards)
        # Obser reward and next obs

        obs, reward, done, infos = eval_envs.step(action)

        # reset num attacks
        scored_mask = reward != 0
        attacks_left[scored_mask] = num_attacks

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()

    print(
        " Evaluation using {} episodes: mean reward {:.5f}, stddev reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards), np.std(eval_episode_rewards)
        )
    )
