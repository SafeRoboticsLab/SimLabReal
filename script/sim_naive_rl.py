# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Naive method training

Run naive methods training, including Base (task rewards only), RP (rewards
with collision penalty), and prior of Recovery RL.

Sample usage: python sim_naive_rl.py -cf config/vanilla/base.yaml
"""

import pretty_errors
import os
import argparse
import wandb
from shutil import copyfile
import matplotlib
from omegaconf import OmegaConf

matplotlib.use('Agg')

from agent import agent_dict
from env import env_dict
from env.vec_env import make_vec_envs
from utils.misc import save_obj


def main(cfg_file, cfg):
    # Config
    os.makedirs(cfg.agent.out_folder, exist_ok=True)
    copyfile(cfg_file, os.path.join(cfg.agent.out_folder, 'cfg.yaml'))
    if cfg.agent.use_wandb:
        wandb.init(
            entity=cfg.agent.entity, project=cfg.agent.project,
            name=cfg.agent.run
        )
        wandb.config.update(cfg)

    # Environment
    print("\n== Environment Information ==")
    env_class = env_dict[cfg.env.name]

    # Make vectorized env
    venv = make_vec_envs(
        env_class,
        cfg.agent.seed,
        cfg.agent.num_cpus,
        cfg.agent.device,
        cfg.agent.cpu_offset,
        cfg.env,
        render=False,
        done_type='fail',
        **cfg.env,
    )
    env = env_class(
        render=False,
        done_type='fail',
        **cfg.env,
    )
    if env.env_type == 'vanilla':
        venv.env_method('report', indices=[0])  # call the method for one env
    venv.reset()
    env.reset()

    # Agent
    print("\n== Agent Information ==")
    agent_class = agent_dict[cfg.agent.name]
    agent = agent_class(cfg.agent, cfg.train, cfg.arch, cfg.env)
    policy = agent.policy
    print(
        '\nTotal parameters in actor: {}'.format(
            sum(
                p.numel() for p in policy.actor.parameters() if p.requires_grad
            )
        )
    )
    print(
        "We want to use: {}, and Agent uses: {}".format(
            cfg.agent.device, policy.device
        )
    )
    print("Critic is using cuda: ", next(policy.critic.parameters()).is_cuda)

    # Learn
    print("\n== Learning ==")
    (train_records, train_progress, violation_record, episode_record) = \
        agent.learn(venv, env, vmin=-0.5, vmax=0.5)

    # Training result dict
    train_dict = {}
    train_dict['train_records'] = train_records
    train_dict['train_progress'] = train_progress
    train_dict['violation_record'] = violation_record
    train_dict['episode_record'] = episode_record
    save_obj(train_dict, os.path.join(cfg.agent.out_folder, 'train'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--config_file", help="config file path", type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    main(args.config_file, cfg)
