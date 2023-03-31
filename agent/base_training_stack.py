# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for training agents using stacked frames as input.

This file implements a parent class for all training agents using stacked
frames.
"""

from typing import Optional, Tuple
from collections import namedtuple
import torch

from agent.replay_memory import ReplayMemoryTraj
from agent.base_training import BaseTraining

TransitionLatentTraj = namedtuple(
    'TransitionLatentTraj', ['z', 's', 'a', 'r', 'done', 'info']
)


class BaseTrainingStack(BaseTraining):

    def __init__(self, cfg, cfg_env, cfg_train, sample_next: bool = False):
        """
        Args:
            cfg (dict): config
            cfg_env (dict): config for environment
            cfg_train (dict): config for training
            sample_next (bool): whether to sample next states when sampling
                batch for policy update. Defaults to False.
        """
        super(BaseTrainingStack, self).__init__(cfg, cfg_env, cfg_train)

        # memory - OVERWRITING
        self.memory = ReplayMemoryTraj(
            cfg.memory_capacity, cfg.seed, sample_next=sample_next
        )
        self.sample_next = sample_next

        # traj
        self.trajs = [[] for _ in range(self.n_envs)]

        # Frame stacking and skipping
        self.traj_size = cfg.traj_size
        self.frame_skip = cfg.frame_skip
        # maxlen in deque; if fewer, randomly sample
        self.traj_cover = (
            self.traj_size - 1
        ) * self.frame_skip + self.traj_size

    def sample_batch_traj(
        self, batch_size: Optional[int] = None, traj_size: int = 50,
        frame_skip: int = 0
    ):
        """Samples batch of trajectories online with assumption of same length.

        Args:
            batch_size (int, optional): batch size. Defaults to None.
            traj_size (int): trajectory length. Defaults to 50.
            frame_skip (int): frame skip. Defaults to 0.
        """
        if batch_size is None:
            batch_size = self.batch_size
        trajs, trajs_nxt = self.memory.sample(
            batch_size, traj_size, frame_skip
        )
        traj_converted_all = []
        for traj in trajs:
            traj_converted = TransitionLatentTraj(*zip(*traj))
            traj_converted_all += [traj_converted]
        if self.sample_next:
            traj_nxt_converted_all = []
            for traj in trajs_nxt:
                if traj == []:  # done
                    traj_converted = []
                else:
                    traj_converted = TransitionLatentTraj(*zip(*traj))
                traj_nxt_converted_all += [traj_converted]
            return traj_converted_all, traj_nxt_converted_all
        else:
            return traj_converted_all

    def store_traj(self, env_ind: int):
        """Stores a trajectory."""
        self.memory.update(self.trajs[env_ind])
        self.trajs[env_ind] = []

    def store_transition_traj(self, env_ind: Optional[int] = None, *args):
        """Stores a transition."""
        self.trajs[env_ind].append(TransitionLatentTraj(*args))

    def unpack_batch(
        self, batch: Tuple, batch_nxt: Tuple, get_perf_action: bool,
        get_latent: bool = False, get_latent_backup: bool = False,
        no_stack: bool = False
    ):
        """Unpacks a batch of trajectories.

        Args:
            batch (list): list of trajectories.
            batch_nxt (list): list of trajectories.
            get_perf_action (bool): whether to get the perfect action.
            get_latent (bool): whether to get the latent. Defaults to False.
            get_latent_backup (bool): whether to get the backup latent.
                Defaults to False.
            no_stack (bool): whether to stack the frames. Defaults to False.
        """
        # Unpack trajs
        traj_all = []
        traj_nxt_all = []
        for traj in batch:
            traj_all += [
                self.unpack_traj(
                    traj, get_perf_action, get_latent, get_latent_backup,
                    no_stack=no_stack
                )
            ]
        for traj_nxt in batch_nxt:
            traj_nxt_all += [
                self.unpack_traj(
                    traj_nxt, get_perf_action, get_latent, get_latent_backup,
                    no_stack=no_stack
                )
            ]

        # (final_flag, obs_stack, reward, g_x, action, append)
        non_final_mask = torch.tensor([not traj[0] for traj in traj_all])
        non_final_state_nxt = torch.cat([
            traj[1].unsqueeze(0) for traj in traj_nxt_all if len(traj) > 0
        ]).to(self.device)
        state = torch.cat([traj[1].unsqueeze(0) for traj in traj_all]
                         ).to(self.device)
        reward = torch.cat([traj[2] for traj in traj_all]).to(self.device)
        g_x = torch.cat([traj[3] for traj in traj_all]).to(self.device)
        action = torch.cat([traj[4] for traj in traj_all]).to(self.device)
        append = torch.cat([traj[5] for traj in traj_all]).to(self.device)
        non_final_append_nxt = torch.cat([
            traj[5] for traj in traj_nxt_all if len(traj) > 0
        ]).to(self.device)

        if get_latent:
            latent = torch.cat([traj[6] for traj in traj_all]).to(self.device)
        else:
            latent = None
        l_x_ra = None
        binary_cost = torch.FloatTensor([
            traj[7]['binary_cost'] for traj in traj_all
        ]).to(self.device)
        hn = None
        cn = None

        return (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x,
            latent, append, non_final_append_nxt, l_x_ra, binary_cost, hn, cn
        )

    def unpack_traj(
        self, traj: TransitionLatentTraj, get_perf_action: bool,
        get_latent: bool, get_latent_backup: bool = False,
        no_stack: bool = False
    ):
        """Unpacks a trajectory. Get state from last step of traj.

        Args:
            traj (TransitionLatentTraj): trajectory.
            get_perf_action (bool): whether to get the action from the
                performance policy.
            get_latent (bool): whether to get the latent.
            get_latent_backup (bool): whether to get the latent used for
                backup. Defaults to False.
            no_stack (bool): whether to stack the frames. Defaults to False.
        """
        if traj == []:
            return ()
        else:
            final_flag = traj.done[-1]
            reward = torch.FloatTensor(traj.r[-1]).to(self.device)
            g_x = torch.FloatTensor([traj.info[-1]['g_x']]).to(self.device)
            if get_perf_action:  # recovery RL separates a_shield and a_perf.
                action = traj.info[-1]['a_perf'].to(self.device)
            else:
                action = traj.a[-1].to(self.device)
            append = traj.info[-1]['append'].to(self.device)
            if get_latent:
                if get_latent_backup:
                    latent = traj.info[-1]['z_backup'].to(self.device)
                else:
                    latent = traj.z[-1].to(self.device)
            else:
                latent = None
            info = traj.info[-1]

            # Stack or not
            obs_seq = torch.cat(traj.s).to(self.device)
            if no_stack:
                obs = obs_seq[-1]
            else:
                obs = torch.flatten(obs_seq, 0, 1)
            return (final_flag, obs, reward, g_x, action, append, latent, info)
