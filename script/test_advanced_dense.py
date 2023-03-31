# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Tests Advanced-Dense environment.

Load a task from the dataset of Advanced-Dense setting, and visualize the
environment in PyBullet.
"""

import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from omegaconf import OmegaConf
from collections import deque
import torch

from agent.SAC_maxEnt import SAC_maxEnt
from agent.SAC_mini import SAC_mini
from env.advanced_dense_env import AdvancedDenseEnv
from utils.misc import check_shielding, get_frames


def main(cfg):

    # Set up performance policy
    perf_policy = SAC_maxEnt(
        cfg.performance.train, cfg.performance.arch, cfg.env
    )
    perf_policy.build_network(build_optimizer=False)

    # Set up backup policy
    backup_policy = SAC_mini(cfg.backup.train, cfg.backup.arch, cfg.env)
    backup_policy.build_network(build_optimizer=False)

    # Load tasks
    with open(cfg.env.dataset, 'rb') as f:
        tasks = pickle.load(f)[0:]

    # Set up a single environment with GUI
    env = AdvancedDenseEnv(
        render=True,
        **cfg.env,
    )

    # Randomly initialize a task
    task = random.choice(tasks)
    obs = env.reset(task=task)
    obs = torch.from_numpy(obs).to(cfg.device)

    # Initialize frame stack
    traj_cover = (cfg.traj_size - 1) * cfg.frame_skip + cfg.traj_size
    prev_obs = deque(maxlen=traj_cover)
    prev_obs.appendleft(obs)

    # Get obs stack
    obs_stack = get_frames(prev_obs, cfg.traj_size,
                           cfg.frame_skip).unsqueeze(0)

    # Sample latent for performance policy
    z = perf_policy.latent_prior.sample().unsqueeze(0)

    # Run steps
    for _ in range(cfg.env.max_step_eval):

        # Get append for performance policy
        append = env.get_append(env._state)
        append = torch.FloatTensor(append).to(cfg.device)

        # Select action
        with torch.no_grad():
            a_exec = torch.zeros((1, 3)).to(cfg.device)
            a_perf, _ = perf_policy.actor.sample(
                obs_stack,
                append=append,
                latent=z,
            )
            a_backup, _ = backup_policy.actor.sample(
                obs_stack,
                append=append,
                latent=None,
            )
            a_exec[0, :-1] = a_perf
            a_exec[0, -1] = perf_policy.act_ind
        a_shield = a_exec.clone()

        # Save robot state for reloading
        _prev_states = np.array(env._state)

        # Shielding
        shield_flag, _ = check_shielding(
            backup_policy,
            cfg.shield_dict,
            obs_stack,
            a_exec,
            append,
            context_backup=None,
            state=_prev_states,
            policy=perf_policy,
            context_policy=z,
        )
        if shield_flag:
            a_backup = backup_policy.actor(
                obs_stack,
                append=append,
                latent=None,
            ).data
            a_shield[0, :-1] = a_backup
            a_shield[0, -1] = backup_policy.act_ind
        a_shield = a_shield[0].cpu().numpy()

        # Step
        obs, reward, done, info = env.step(a_shield)
        print(
            f"Distance to goal: {info['l_x']}; distance to obstacle: {info['g_x']}; heading: {info['heading']}; reward: {reward}; done: {done}"
        )
        plt.imshow(obs[:3, :, :].transpose(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--config_file", help="config file path", type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    main(cfg)
