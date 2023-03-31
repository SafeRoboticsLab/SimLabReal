# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for naive policy training with frame stacking.

This file implements the training of the naive policy methods, including Base
(task rewards only), RP (rewards with penalty), and prior of Recovery RL. The
policy has stacked frames as input.
"""

from typing import Optional

import os
import time
import numpy as np
import torch
import wandb
from collections import deque

from agent.SAC_mini import SAC_mini
from agent.base_training_stack import BaseTrainingStack
from utils.misc import get_frames
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class NaiveStackRL(BaseTrainingStack):

    def __init__(
        self, cfg, cfg_train, cfg_arch, cfg_env, verbose: bool = True
    ):
        """
        Args:
            cfg: config for agent
            cfg_train: config for policy training
            cfg_arch: config for NN architecture
            cfg_env: config for environment
            verbose: whether to print out info
        """
        super().__init__(cfg, cfg_env, cfg_train, sample_next=True)

        print("= Constructing policy agent")
        self.policy = SAC_mini(cfg_train, cfg_arch, cfg_env)
        self.policy.build_network(
            actor_path=cfg_train.actor_path, critic_path=cfg_train.critic_path,
            verbose=verbose
        )

        # alias
        self.module_all = [self.policy]
        self.performance = self.policy

    @property
    def has_backup(self):
        return False

    def reset_all_envs(self, venv: VecEnvWrapper, random_init: bool = False):
        """Reset all environments

        Args:
            venv: vectorized environment
            random_init: whether to randomly initialize the robot pose
        """
        s, _ = venv.reset(random_init=random_init)

        # Initialize prev_obs - add initial one
        _prev_obs = [deque(maxlen=self.traj_cover) for _ in range(self.n_envs)]
        for s_env, _prev_obs_env in zip(s, _prev_obs):
            _prev_obs_env.appendleft(s_env)

        # Get obs stack
        _prev_obs_stack = []
        for _prev_obs_env in _prev_obs:
            _obs_stack_env = get_frames(
                _prev_obs_env, self.traj_size, self.frame_skip
            ).unsqueeze(0)
            _prev_obs_stack += [_obs_stack_env]
        _prev_obs_stack = torch.cat(_prev_obs_stack)

        return s, _prev_obs, _prev_obs_stack

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ):
        """
        Learns the naive RL policy with stacked observations as input: single
        mode and without latent.

        Args:
            venv (VecEnvWrapper):  vectorized environment.
            env (BaseEnv): a single environment for evaluation.
            current_step (Optional[int], optional): current number of training
                steps. Defaults to None.
            vmin (float, optional): minimum value for plotting. Defaults to -1.
            vmax (float, optional): maximum value for plotting. Defaults to 1.
            save_figure (bool, optional): saves figures if True. Defaults to
                True.
            plot_figure (bool, optional): plots figures if True. Defaults to
                False.

        Returns:
            np.ndarray: training loss at every record cycle.
            np.ndarray: training progress at every record cycle.
            np.ndarray: safety violations so far at every record cycle.
            np.ndarray: experienced episodes so far at every record cycle.
        """
        # hyper-parameters
        random_init = self.cfg.random_init
        max_steps = self.cfg.max_steps
        opt_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
        min_step_b4_opt = self.cfg.min_steps_b4_opt

        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder
        if self.cfg.check_type == 'all_env':
            num_traj_per_env = self.cfg.num_traj_per_env
            num_rnd_traj = num_traj_per_env * venv.num_task
        else:
            num_rnd_traj = self.cfg.num_validation_trajs

        # == Main Training ==
        start_learning = time.time()
        train_records = []
        train_progress = [[], []]
        violation_record = []
        episode_record = []
        cnt_opt = 0
        cnt_opt_period = 0
        cnt_safety_violation = 0
        cnt_num_episode = 0

        # Saves model
        model_folder = os.path.join(out_folder, 'model')
        os.makedirs(model_folder, exist_ok=True)
        self.module_folder_all = [model_folder]
        save_metric = self.cfg.save_metric

        if save_figure:
            figure_folder = os.path.join(out_folder, 'figure')
            os.makedirs(figure_folder, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Resets all envs
        s, _prev_obs, _prev_obs_stack = self.reset_all_envs(
            venv, random_init=random_init
        )

        # Steps
        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append - add noise
            append_all = venv.get_append(venv.get_attr('_state'))
            if self.use_append_noise:
                append_noise = self.append_noise_dist.sample((self.n_envs,))
                append_all += append_noise

            # Selects action
            with torch.no_grad():
                a_all = torch.zeros((self.n_envs, self.action_dim + 1))
                a_all = a_all.to(self.device)
                a_tmp, _ = self.policy.actor.sample(
                    _prev_obs_stack, append=append_all, latent=None
                )
                a_all[:, :-1] = a_tmp.clone()
                a_all[:, -1] = self.policy.act_ind

            # Applies action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_all)

            # Add to prev_obs
            for s_env, _prev_obs_env in zip(s, _prev_obs):
                _prev_obs_env.appendleft(s_env)

            # Gets new append
            append_nxt_all = venv.get_append(venv.get_attr('_state'))

            # Get new obs_stack
            _obs_stack = []
            for _prev_obs_env in _prev_obs:
                _obs_stack_env = get_frames(
                    _prev_obs_env, self.traj_size, self.frame_skip
                ).unsqueeze(0)
                _obs_stack += [_obs_stack_env]
            _obs_stack = torch.cat(_obs_stack)

            # Checkss all envs
            for env_ind, (_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):

                # Saves append
                info['append'] = append_all[env_ind].unsqueeze(0)
                info['append_nxt'] = append_nxt_all[env_ind].unsqueeze(0)

                # Stores the transition in memory
                if self.policy.critic_has_act_ind:
                    action = a_all[env_ind].unsqueeze(0).clone()
                else:
                    action = a_all[env_ind, :-1].unsqueeze(0).clone()
                self.store_transition_traj(
                    env_ind, None,
                    s[env_ind].unsqueeze(0).to(self.image_device), action, r,
                    done, info
                )

                if done:
                    # Pushes traj to replay buffer and reset traj buffer for
                    # that env
                    self.store_traj(env_ind)

                    # Resets environment with randomly sampled task
                    obs, _ = venv.reset_one(
                        env_ind, random_init=random_init, verbose=False
                    )
                    _prev_obs[env_ind] = deque(maxlen=self.traj_cover)

                    # Updates observations
                    s_all[env_ind] = obs
                    _prev_obs[env_ind].appendleft(obs)

                    # Checks safety violation
                    g_x = info['g_x']
                    if g_x > 0:
                        cnt_safety_violation += 1
                    cnt_num_episode += 1
            violation_record.append(cnt_safety_violation)
            episode_record.append(cnt_num_episode)

            # Updatess "prev" states
            s = s_all
            _prev_obs_stack = _obs_stack

            # Optimizes
            if (
                self.cnt_step >= min_step_b4_opt and cnt_opt_period >= opt_freq
            ):
                cnt_opt_period = 0

                # Updates critic/actor
                loss = np.zeros(4)
                self.memory.set_possible_samples(
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                    allow_repeat_frame=False,
                )
                for timer in range(num_update_per_opt):
                    _batch, _batch_nxt = self.sample_batch_traj(
                        batch_size=self.batch_size,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    batch = self.unpack_batch(
                        _batch, _batch_nxt, get_perf_action=False
                    )
                    loss_tp = self.policy.update(
                        batch, timer, update_period=self.update_period
                    )
                    for i, l in enumerate(loss_tp):
                        loss[i] += l
                loss /= num_update_per_opt

                # Records loss_q, loss_pi, loss_entropy, loss_alpha
                train_records.append(loss)
                if self.cfg.use_wandb:
                    wandb.log({
                        "loss_q": loss[0],
                        "loss_pi": loss[1],
                        "loss_entropy": loss[2],
                        "loss_alpha": loss[3],
                    }, step=self.cnt_step, commit=False)

                # Re-initializes pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, _prev_obs, _prev_obs_stack = self.reset_all_envs(
                    venv, random_init=random_init
                )

                # Counts number of optimization
                cnt_opt += 1

                # Checks after fixed number of steps
                if cnt_opt % check_opt_freq == 0:
                    print(
                        '\n  - Safety violations so far: {:d}'.
                        format(cnt_safety_violation)
                    )
                    if self.use_append_noise:
                        print('  - nu={:.2f}'.format(self.nu))
                    print('  - buffer size={:d}'.format(len(self.memory)))
                    print(
                        '  - buffer sample={:d}'.format(
                            self.memory.num_sample
                        )
                    )

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Sets states for check()
                    sample_states = self.get_check_states(env, num_rnd_traj)
                    progress = self.policy.check(
                        venv,
                        self.cnt_step,
                        states=sample_states,
                        check_type='random',
                        verbose=True,
                        revert_task=True,
                        sample_task=True,
                        num_rnd_traj=num_rnd_traj,
                        # kwargs for env.simulate_trajectories
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    train_progress[0].append(progress)
                    train_progress[1].append(self.cnt_step)
                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success": progress[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
                            "buffer traj": len(self.memory),
                            "buffer sample": self.memory.num_sample,
                        }, step=self.cnt_step, commit=True)
                        if self.use_append_noise:
                            wandb.log({
                                "NU": self.nu,
                            }, step=self.cnt_step, commit=True)

                    # Saves model
                    if save_metric == 'perf':
                        self.save(metric=progress[0])
                    else:
                        raise NotImplementedError

                    # Saves training details
                    torch.save({
                        'train_records': train_records,
                        'train_progress': train_progress,
                        'violation_record': violation_record,
                        "episode_record": episode_record,
                    }, os.path.join(out_folder, 'train_details'))

                    if plot_figure or save_figure:
                        self.get_figures(
                            venv,
                            env,
                            plot_v,
                            vmin,
                            vmax,
                            save_figure,
                            plot_figure,
                            figure_folder,
                            plot_backup=False,
                            figure_folder_backup=None,
                            plot_shield=False,
                            figure_folder_shield=None,
                            shield_dict=None,
                            traj_size=self.traj_size,
                            frame_skip=self.frame_skip,
                        )

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Re-initializes env
                    env.close_pb()
                    env.reset()

            # Counts
            self.cnt_step += self.n_envs
            cnt_opt_period += self.n_envs

            # Updates gamma, lr etc.
            for _ in range(self.n_envs):
                self.policy.update_hyper_param()
                if self.use_append_noise:
                    self.nu_scheduler.step()
            if self.use_append_noise:
                self.nu = self.nu_scheduler.get_variable()
                self.append_noise_dist = torch.distributions.Normal(
                    torch.zeros((2), device=self.device),
                    self.nu * self.append_noise_std
                )

        self.save(force_save=True)
        end_learning = time.time()
        time_learning = end_learning - start_learning
        print('\nLearning: {:.1f}'.format(time_learning))

        train_records = np.array(train_records)
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        violation_record = np.array(violation_record)
        episode_record = np.array(episode_record)
        return train_records, train_progress, violation_record, episode_record
