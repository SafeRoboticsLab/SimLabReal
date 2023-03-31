# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Tests Vanilla-Dynamics environment.

Loads a task from the dataset of Vanilla-Dynamics setting, tests the proposed
PAC-Shield-Perf, and visualizes an environment in PyBullet.
"""

import argparse
from omegaconf import OmegaConf
import torch
import numpy as np
import random
import time

from agent.deployment_agent import DeploymentAgent
from env.vec_env import make_vec_envs
from env.vanilla_env import VanillaEnv
from utils.misc import check_shielding


def main(cfg):
    # Loads tasks and constructs env and venv.
    env = VanillaEnv(**cfg.env, render=cfg.env.gui)
    env.done_type = 'TF'
    env.reset()
    venv = make_vec_envs(
        VanillaEnv, cfg.agent.seed, cfg.agent.num_cpus, cfg.agent.device,
        cfg.agent.cpu_offset, cfg.env, num_gpu_envs=cfg.agent.num_gpu_envs,
        render=False, **cfg.env
    )
    venv.reset()
    env.report()
    tasks = venv.task_all

    # Loads agents.
    print("= Constructing performance agent")
    performance = DeploymentAgent(
        cfg.performance.train, cfg.performance.arch, cfg.env
    )
    performance.build_network()
    if performance.has_latent:
        latent_dist = performance.latent_dist
    else:
        latent_dist = None

    if cfg.agent.has_backup:
        print("= Constructing backup agent")
        backup = DeploymentAgent(cfg.backup.train, cfg.backup.arch, cfg.env)
        backup.build_network()
        shield = True
        shield_dict = cfg.agent.shield_dict
    else:
        backup = None
        shield = False
        shield_dict = None

    # Evaluates.
    performance.check(
        env=venv, cnt_step=0, states=None, check_type='all_env', verbose=True,
        num_traj_per_env=1, revert_task=True, latent_dist=latent_dist,
        shield=shield, backup=backup, shield_dict=shield_dict, fixed_init=True
    )

    # Visualizes.
    if cfg.env.gui:
        if hasattr(cfg.env, 'gui_task_idx'):
            task_idx = cfg.env.gui_task_idx
        else:
            task_idx = random.randint(len(tasks))
        task = tasks[task_idx]
        obs = env.reset(task=task)

        if performance.has_latent:
            z = performance.latent_dist.sample().unsqueeze(0)
        else:
            z = None

        for t in range(cfg.env.max_step_eval):
            obs = torch.from_numpy(obs).to(cfg.agent.device).unsqueeze(0)

            # Gets append for performance policy
            append = env.get_append(env._state)
            append = torch.FloatTensor(append).to(cfg.agent.device)

            # Selects action
            with torch.no_grad():
                a_exec = torch.zeros((1, 3)).to(cfg.agent.device)
                a_perf = performance.actor(obs, append=append, latent=z)
                a_exec[0, :-1] = a_perf
                a_exec[0, -1] = performance.act_ind
            a_shield = a_exec.clone()

            # Saves robot state for reloading.
            _prev_state = np.array(env._state)

            # Shields.
            if cfg.agent.has_backup:
                shield_flag, _ = check_shielding(
                    backup, shield_dict, obs, a_exec, append,
                    context_backup=None, state=_prev_state, policy=performance,
                    context_policy=z
                )
                if shield_flag:
                    with torch.no_grad():
                        a_backup = backup.actor(
                            obs, append=append, latent=None
                        )
                    a_shield[0, :-1] = a_backup
                    a_shield[0, -1] = backup.act_ind
            a_shield = a_shield[0].cpu().numpy()

            # Steps.
            obs, reward, done, info = env.step(a_shield)

            # Logs.
            print(
                f"[{t:d}] Distance to goal: {info['l_x']}; "
                + f"distance to obstacle: {info['g_x']}; "
                + f"reward: {reward}; done: {done}"
            )
            # plt.imshow(obs[:3, :, :].transpose(1, 2, 0))
            # plt.show()
            if info['g_x'] > 0.:
                print("Collision!")
                break
            if info['l_x'] < 0.:
                print("Reach goal!")
                time.sleep(0.5)
                break
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--config_file", help="config file path", type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    main(cfg)
