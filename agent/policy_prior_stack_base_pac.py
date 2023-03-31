# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training prior for PAC-Base and PAC-RP with frame stacking.

This file implements the training of the prior policy with latent-conditioned
performance policy without backup policy or shielding, which are the methods
PAC-Base and PAC-RP in the paper. Additionally, the performance policy has
stacked frames as input.

The performance policy is trained with task rewards (with penalty for hitting
the obstacle in PAC-RP) and diversity reward that maximizes mutual information
(MI) between the state and the latent (observation-marginal MI) implemented
in SAC_maxEnt.py.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import torch
import wandb
from collections import deque

from agent.base_training_stack import BaseTrainingStack
from agent.SAC_maxEnt import SAC_maxEnt
from utils.misc import get_frames
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class PolicyPriorStackBasePAC(BaseTrainingStack):

    def __init__(self, cfg, cfg_performance, cfg_env, verbose: bool = True):
        """
        Args:
            cfg: config for agent
            cfg_performance: config for performance policy
            cfg_backup: config for backup policy
            cfg_env: config for environment
            verbose: whether to print verbose info
        """
        super().__init__(cfg, cfg_env, cfg_performance.train, sample_next=True)
        self.disc_batch_size = cfg_performance.train.disc_batch_size
        self.disc_recent_size = cfg_performance.train.disc_recent_size

        print("= Constructing performance agent")
        self.performance = SAC_maxEnt(
            cfg_performance.train, cfg_performance.arch, cfg_env
        )
        self.performance.build_network(verbose=verbose)

        # For saving and removing models
        self.module_all = [self.performance]

    @property
    def has_backup(self):
        return False

    def reset_all_envs(self, venv: VecEnvWrapper, random_init: bool = False):
        """Resets all environments

        Args:
            venv (VecEnvWrapper):  vectorized environment.
            random_init (bool): whether to randomly initialize the robot pose.
                Defaults to False.
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the prior of PAC-Base and PAC-RP with frame stacking.

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
        disc_num_update_per_opt = self.cfg.disc_update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
        min_step_b4_opt = self.cfg.min_steps_b4_opt
        num_rnd_traj = self.cfg.num_validation_trajs
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder

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
        model_folder_perf = os.path.join(model_folder, 'performance')
        model_folder_disc = os.path.join(model_folder_perf, 'disc')
        os.makedirs(model_folder_perf, exist_ok=True)
        os.makedirs(model_folder_disc, exist_ok=True)
        self.module_folder_all = [model_folder_perf]
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
        z = self.performance.latent_prior.sample((self.n_envs,))
        z = z.to(self.device)

        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append for performance policy
            append = venv.get_append(venv.get_attr('_state'))
            if self.use_append_noise:
                append_noise = self.append_noise_dist.sample((self.n_envs,))
                append += append_noise

            # Selects action - random if before 1st optimize
            with torch.no_grad():
                a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                a_exec = a_exec.to(self.device)
                a_perf, _ = self.performance.actor.sample(
                    _prev_obs_stack, append=append, latent=z
                )
                a_exec[:, :-1] = a_perf
                a_exec[:, -1] = self.performance.act_ind

            # Applies action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_exec)

            # Add to prev_obs
            for s_env, _prev_obs_env in zip(s, _prev_obs):
                _prev_obs_env.appendleft(s_env)

            # Gets new append
            append_next = venv.get_append(venv.get_attr('_state'))

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
                info['append'] = append[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next[env_ind].unsqueeze(0)
                info['a_perf'] = a_exec[env_ind, :-1].unsqueeze(0)

                # Stores the transition in memory
                self.store_transition_traj(
                    env_ind, z[env_ind].clone().unsqueeze(0),
                    s[env_ind].unsqueeze(0).to(self.image_device),
                    a_exec[env_ind].unsqueeze(0), r, done, info
                )

                # Resets if failure or timeout
                if done:
                    # Push traj to replay buffer and reset traj buffer for
                    # that env.
                    self.store_traj(env_ind)

                    # Re-sample z
                    z[env_ind] = self.performance.latent_prior.sample().to(
                        self.device
                    )

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
                loss_perf = np.zeros(5)  # disc

                # Updates discriminator
                loss_disc = []
                self.memory.set_possible_samples(
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                    allow_repeat_frame=False,
                    recent_size=self.disc_recent_size,
                )
                for _ in range(disc_num_update_per_opt):
                    _batch, _batch_nxt = self.sample_batch_traj(
                        batch_size=self.disc_batch_size,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    online_batch = self.unpack_batch(
                        _batch, _batch_nxt, get_perf_action=False,
                        get_latent=True
                    )
                    loss_disc += [self.performance.update_disc(online_batch)]
                loss_disc = np.mean(loss_disc)

                # Updates critic/actor
                self.memory.set_possible_samples(
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                    allow_repeat_frame=False,
                    recent_size=None,  # Resets
                )
                for timer in range(num_update_per_opt):
                    _batch, _batch_nxt = self.sample_batch_traj(
                        batch_size=self.batch_size,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    batch_perf = self.unpack_batch(
                        _batch, _batch_nxt, get_perf_action=True,
                        get_latent=True
                    )
                    loss_tp_perf = self.performance.update(
                        batch_perf, timer, update_period=self.update_period
                    )
                    for i, l in enumerate(loss_tp_perf):
                        loss_perf[i] += l
                loss_perf /= num_update_per_opt
                loss_perf[-1] = loss_disc

                # Record
                train_records.append(loss_perf)
                if self.cfg.use_wandb:
                    lr = self.performance.critic_optimizer.state_dict(
                    )['param_groups'][0]['lr']
                    wandb.log({
                        "loss_q (Perf)": loss_perf[0],
                        "loss_pi (Perf)": loss_perf[1],
                        "loss_entropy (Perf)": loss_perf[2],
                        "loss_alpha (Perf)": loss_perf[3],
                        "loss_disc (Perf)": loss_perf[4],
                        "learning rate (Perf)": lr
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
                    progress_perf = self.performance.check(
                        venv,
                        self.cnt_step,
                        states=sample_states,
                        check_type='random',
                        verbose=True,
                        revert_task=True,
                        sample_task=True,
                        num_rnd_traj=num_rnd_traj,
                        # kwargs for env.simulate_trajectories
                        latent_dist=self.performance.latent_prior,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    train_progress[0].append(progress_perf)
                    train_progress[1].append(self.cnt_step)
                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success": progress_perf[0],
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
                        self.save(metric=progress_perf[0])
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
                self.performance.update_hyper_param()
                if self.performance.latent_std_scheduler is not None:
                    self.performance.latent_std_scheduler.step()
                if self.use_append_noise:
                    self.nu_scheduler.step()
            if self.use_append_noise:
                self.nu = self.nu_scheduler.get_variable()
                self.append_noise_dist = torch.distributions.Normal(
                    torch.zeros((2), device=self.device),
                    self.nu * self.append_noise_std
                )
            if self.performance.latent_std_scheduler is not None:
                self.performance.update_latent()
            if self.performance.aug_reward_range_scheduler is not None:
                self.performance.update_aug_reward_range()

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
