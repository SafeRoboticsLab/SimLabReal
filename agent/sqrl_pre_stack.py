# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training prior for SQRL with frame stacking.

This file implements the training of a performance policy and a backup policy,
which is the method SQRL in the paper.

The performance policy is trained with task rewards. The backup policy is
trained with binary collision indicators (sparse rewards). See SAC_sqrl.py for
the details.

Paper: https://arxiv.org/abs/2010.14603.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import torch
import wandb
from copy import deepcopy
from collections import deque

from agent.base_training_stack import BaseTrainingStack, TransitionLatentTraj
from agent.replay_memory import ReplayMemoryTraj
from agent.SAC_mini import SAC_mini
from agent.SAC_sqrl import SAC_sqrl
from utils.misc import check_shielding, get_frames
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class SQRLPreStack(BaseTrainingStack):

    def __init__(
        self, cfg, cfg_performance, cfg_backup, cfg_env, verbose: bool = True
    ):
        """
        Args:
            cfg: config for agent
            cfg_performance: config for performance policy
            cfg_backup: config for backup policy
            cfg_env: config for environment
            verbose: whether to print verbose info
        """
        super().__init__(cfg, cfg_env, cfg_performance.train, sample_next=True)

        print("= Constructing performance agent")
        self.performance = SAC_mini(
            cfg_performance.train, cfg_performance.arch, cfg_env
        )
        self.performance.build_network(verbose=verbose)

        print("= Constructing backup agent")
        self.backup = SAC_sqrl(cfg_backup.train, cfg_backup.arch, cfg_env)
        self.backup.build_network(verbose=verbose)

        # the first stage in SQRL needs to have two separate buffers
        self.memory_online = ReplayMemoryTraj(
            cfg.on_memory_capacity, cfg.seed, sample_next=True
        )

        # For saving and removing models
        self.module_all = [self.performance, self.backup]

    @property
    def has_backup(self):
        return True

    def reset_all_envs(self, venv: VecEnvWrapper, random_init: bool = False):
        """Resets all environments

        Args:
            venv (VecEnvWrapper):  vectorized environment.
            random_init (bool): whether to randomly initialize the robot pose.
                Defaults to False.
        """
        s, _ = venv.reset(random_init=random_init)

        # since shared between two replay buffers
        self.trajs = [[] for _ in range(self.n_envs)]

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
        """Learns the prior of SQRL with frame stacking.

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
        perf_steps = self.cfg.perf_steps
        backup_steps = self.cfg.backup_steps
        opt_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        min_step_b4_opt = self.cfg.min_steps_b4_opt
        if self.cfg.check_type == 'all_env':
            num_traj_per_env = self.cfg.num_traj_per_env
            num_rnd_traj = num_traj_per_env * venv.num_task
        else:
            num_rnd_traj = self.cfg.num_validation_trajs
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder
        shield_dict = self.cfg.train_shield_dict

        # == Main Training ==
        start_learning = time.time()
        train_records = [[], []]
        train_progress = [[], [], [], []]
        violation_record = []
        episode_record = []
        cnt_safety_violation = 0
        cnt_num_episode = 0
        cnt_perf_steps = 0
        cnt_backup_steps = 0

        # Saves model
        model_folder = os.path.join(out_folder, 'model')
        model_folder_perf = os.path.join(model_folder, 'performance')
        model_folder_backup = os.path.join(model_folder, 'backup')
        os.makedirs(model_folder_perf, exist_ok=True)
        os.makedirs(model_folder_backup, exist_ok=True)
        self.module_folder_all = [model_folder_perf, model_folder_backup]
        save_metric = self.cfg.save_metric

        if save_figure:
            figure_folder = os.path.join(out_folder, 'figure')
            figure_folder_perf = os.path.join(figure_folder, 'performance')
            figure_folder_backup = os.path.join(figure_folder, 'backup')
            figure_folder_shield = os.path.join(figure_folder, 'shield')
            os.makedirs(figure_folder_perf, exist_ok=True)
            os.makedirs(figure_folder_backup, exist_ok=True)
            os.makedirs(figure_folder_shield, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        while self.cnt_step < max_steps:
            # Use performance policy for the first part of the cycle
            s, _prev_obs, _prev_obs_stack = self.reset_all_envs(
                venv, random_init=random_init
            )
            cnt_opt_period = 0
            print(self.cnt_step, end='\r')

            while cnt_perf_steps < perf_steps:
                # Saves robot state for reloading
                _prev_states = np.array(venv.get_attr('_state'))

                # Setss train modes for all envs
                venv.env_method('set_train_mode')

                # Gets append
                append = venv.get_append(_prev_states)
                if self.use_append_noise:
                    append_noise = self.append_noise_dist.sample(
                        (self.n_envs,)
                    )
                    append += append_noise

                # Selects action
                with torch.no_grad():
                    a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                    a_exec = a_exec.to(self.device)
                    a_perf, _ = self.performance.actor.sample(
                        _prev_obs_stack, append=append, latent=None
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
                        env_ind, None,
                        s[env_ind].unsqueeze(0).to(self.image_device),
                        a_exec[env_ind].unsqueeze(0), r, done, info
                    )

                    # Resets if failure or timeout
                    if done:
                        # Pushes traj to replay buffer and reset traj buffer
                        # for that env.
                        self.store_traj(env_ind)

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

                # Counts
                self.cnt_step += self.n_envs
                cnt_opt_period += self.n_envs
                if cnt_perf_steps < perf_steps:
                    cnt_perf_steps += self.n_envs

                # Optimizes
                if (
                    cnt_opt_period >= opt_freq
                    and cnt_perf_steps >= min_step_b4_opt
                ):
                    print("Performance optimization starts")
                    cnt_opt_period = 0
                    loss_perf = np.zeros(4)

                    # Updates critic/actor
                    self.memory.set_possible_samples(
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                        allow_repeat_frame=False,
                        recent_size=None,
                    )
                    for timer in range(num_update_per_opt):
                        # update performance using offline replay buffer
                        _batch, _batch_nxt = self.sample_batch_traj(
                            batch_size=self.batch_size,
                            traj_size=self.traj_size,
                            frame_skip=self.frame_skip,
                        )
                        batch_perf = self.unpack_batch(
                            _batch, _batch_nxt, get_perf_action=True
                        )

                        loss_tp_perf = self.performance.update(
                            batch_perf, timer, update_period=self.update_period
                        )
                        for i, l in enumerate(loss_tp_perf):
                            loss_perf[i] += l
                    loss_perf /= num_update_per_opt

                    # Records loss_q, loss_pi, loss_entropy, loss_alpha
                    train_records[0].append(loss_perf)
                    if self.cfg.use_wandb:
                        lr = self.performance.critic_optimizer.state_dict(
                        )['param_groups'][0]['lr']
                        wandb.log({
                            "loss_q (Perf)": loss_perf[0],
                            "loss_pi (Perf)": loss_perf[1],
                            "loss_entropy (Perf)": loss_perf[2],
                            "loss_alpha (Perf)": loss_perf[3],
                            "learning rate (Perf)": lr
                        }, step=self.cnt_step, commit=False)

                # Updates gamma, lr etc.
                for _ in range(self.n_envs):
                    self.performance.update_hyper_param()
                    if self.use_append_noise:
                        self.nu_scheduler.step()
                if self.use_append_noise:
                    self.nu = self.nu_scheduler.get_variable()
                    self.append_noise_dist = torch.distributions.Normal(
                        torch.zeros((2), device=self.device),
                        self.nu * self.append_noise_std
                    )

            # Use rejection sampling for the second part of the cycle
            init_flag_all = torch.full(size=(self.n_envs,), fill_value=True)
            s_old = None
            a_old = None
            done_old = None
            r_old = None
            info_old = None
            append_old = None
            s, _prev_obs, _prev_obs_stack = self.reset_all_envs(
                venv, random_init=random_init
            )
            cnt_opt_period = 0
            print("On-Policy Samples")
            while cnt_backup_steps < backup_steps:

                # Saves robot state for reloading
                _prev_states = np.array(venv.get_attr('_state'))

                # Setss train modes for all envs
                venv.env_method('set_train_mode')

                # Gets append
                append = venv.get_append(_prev_states)
                if self.use_append_noise:
                    append_noise = self.append_noise_dist.sample(
                        (self.n_envs,)
                    )
                    append += append_noise

                # Selects action
                with torch.no_grad():
                    a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                    a_exec = a_exec.to(self.device)
                    a_perf, _ = self.performance.actor.sample(
                        _prev_obs_stack, append=append, latent=None
                    )
                    a_exec[:, :-1] = a_perf
                    a_exec[:, -1] = self.performance.act_ind
                a_shield = a_exec.clone()

                # Shielding
                _, shield_info = check_shielding(
                    self.backup, shield_dict, _prev_obs_stack, a_shield,
                    append, context_backup=None, state=_prev_states,
                    policy=self.performance, context_policy=None
                )
                a_shield = shield_info['action_final']

                # Applies action - update heading
                s_all, r_all, done_all, info_all = venv.step(a_shield)

                # Add to prev_obs
                for s_env, _prev_obs_env in zip(s, _prev_obs):
                    _prev_obs_env.appendleft(s_env)

                # Get new obs_stack
                _obs_stack = []
                for _prev_obs_env in _prev_obs:
                    _obs_stack_env = get_frames(
                        _prev_obs_env, self.traj_size, self.frame_skip
                    ).unsqueeze(0)
                    _obs_stack += [_obs_stack_env]
                _obs_stack = torch.cat(_obs_stack)

                # Checks envs
                for env_ind, (done, init_flag) in enumerate(
                    zip(done_all, init_flag_all)
                ):
                    # Stores the transition in memory
                    if not init_flag:
                        info = deepcopy(info_old[env_ind])
                        info['a_'] = a_shield[env_ind].unsqueeze(0)
                        info['append'] = append_old[env_ind].unsqueeze(0)
                        info['append_nxt'] = append[env_ind].unsqueeze(0)
                        self.store_transition_traj(
                            env_ind, None,
                            s_old[env_ind].unsqueeze(0).to(self.image_device),
                            a_old[env_ind].unsqueeze(0), r_old[env_ind],
                            done_old[env_ind], info
                        )

                    # Resets if failure or timeout
                    if done:
                        self.store_transition_traj(
                            env_ind, None,
                            s_all[env_ind].unsqueeze(0).to(self.image_device),
                            a_shield[env_ind].unsqueeze(0), r_all[env_ind],
                            True, info
                        )

                        # Pushes traj to replay buffer and reset traj buffer
                        # for that env.
                        self.store_traj_online(env_ind)

                        # Resets environment with randomly sampled task
                        obs, _ = venv.reset_one(
                            env_ind, random_init=random_init, verbose=False
                        )
                        _prev_obs[env_ind] = deque(maxlen=self.traj_cover)

                        # Updates observations
                        s_all[env_ind] = obs
                        _prev_obs[env_ind].appendleft(obs)

                        init_flag_all[env_ind] = True
                        g_x = info['g_x']
                        if g_x > 0:
                            cnt_safety_violation += 1
                        cnt_num_episode += 1
                    else:
                        init_flag_all[env_ind] = False
                violation_record.append(cnt_safety_violation)
                episode_record.append(cnt_num_episode)

                # Updatess "prev" states
                s_old = s.clone()
                a_old = a_shield.clone()
                r_old = r_all.clone()
                done_old = done_all.copy()
                info_old = deepcopy(info_all)
                append_old = append.clone()
                #
                s = s_all
                _prev_obs_stack = _obs_stack

                # Counts
                self.cnt_step += self.n_envs
                cnt_opt_period += self.n_envs
                if cnt_backup_steps < backup_steps:
                    cnt_backup_steps += self.n_envs

                # Optimizes
                if (cnt_opt_period >= opt_freq):
                    print("Backup optimization starts")
                    cnt_opt_period = 0
                    loss_backup = np.zeros(4)

                    # SQRL does not use the backup actor.
                    self.memory_online.set_possible_samples(
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                        allow_repeat_frame=False,
                        recent_size=None,
                    )
                    for timer in range(num_update_per_opt):
                        # update backup using online replay buffer
                        _batch, _batch_nxt = self.sample_batch_traj_online(
                            batch_size=self.batch_size,
                            traj_size=self.traj_size,
                            frame_skip=self.frame_skip,
                        )
                        batch_backup = self.unpack_batch_backup(
                            _batch, _batch_nxt, get_perf_action=False
                        )

                        loss_tp_backup = self.backup.update(
                            batch_backup, timer,
                            update_period=self.update_period
                        )
                        for i, l in enumerate(loss_tp_backup):
                            loss_backup[i] += l
                    loss_backup /= num_update_per_opt

                    # Records loss_q, loss_pi, loss_entropy, loss_alpha
                    train_records[1].append(loss_backup)
                    if self.cfg.use_wandb:
                        lr = self.performance.critic_optimizer.state_dict(
                        )['param_groups'][0]['lr']
                        wandb.log({
                            "loss_q (Backup)": loss_backup[0],
                            "loss_pi (Backup)": loss_backup[1],
                            "loss_entropy (Backup)": loss_backup[2],
                            "loss_alpha (Backup)": loss_backup[3],
                        }, step=self.cnt_step, commit=False)

                # Updates gamma, lr etc.
                for _ in range(self.n_envs):
                    self.backup.update_hyper_param()
                    if self.use_append_noise:
                        self.nu_scheduler.step()
                if self.use_append_noise:
                    self.nu = self.nu_scheduler.get_variable()
                    self.append_noise_dist = torch.distributions.Normal(
                        torch.zeros((2), device=self.device),
                        self.nu * self.append_noise_std
                    )

            # Resets counter of steps
            cnt_perf_steps = 0
            cnt_backup_steps = 0

            # Checks after a cycle
            print(
                '\n  - Safety violations so far: {:d}'.
                format(cnt_safety_violation)
            )
            print('  - buffer size={:d}'.format(len(self.memory)))
            print('  - buffer sample={:d}'.format(self.memory.num_sample))

            # Releases GPU RAM as much as possible
            torch.cuda.empty_cache()

            # Sets states for check()
            sample_states = self.get_check_states(env, num_rnd_traj)
            if self.cfg.check_type == 'all_env':
                progress_perf = self.performance.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='all_env',
                    verbose=True,
                    num_traj_per_env=num_traj_per_env,
                    revert_task=True,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
                progress_backup = self.backup.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='all_env',
                    verbose=True,
                    num_traj_per_env=num_traj_per_env,
                    revert_task=True,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
                progress_perf_shield_rej = self.performance.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='all_env',
                    verbose=True,
                    num_traj_per_env=num_traj_per_env,
                    revert_task=True,
                    # kwargs for env.simulate_trajectories
                    shield=True,
                    backup=self.backup,
                    shield_dict=shield_dict,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
            else:
                progress_perf = self.performance.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='random',
                    verbose=True,
                    revert_task=True,
                    sample_task=True,
                    num_rnd_traj=num_rnd_traj,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
                progress_backup = self.backup.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='random',
                    verbose=True,
                    revert_task=True,
                    sample_task=True,
                    num_rnd_traj=num_rnd_traj,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
                progress_perf_shield_rej = self.performance.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='random',
                    verbose=True,
                    revert_task=True,
                    sample_task=True,
                    num_rnd_traj=num_rnd_traj,
                    # kwargs for env.simulate_trajectories
                    shield=True,
                    backup=self.backup,
                    shield_dict=shield_dict,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )
            train_progress[0].append(progress_perf)
            train_progress[1].append(progress_backup)
            train_progress[2].append(progress_perf_shield_rej)
            train_progress[3].append(self.cnt_step)
            if self.cfg.use_wandb:
                wandb.log({
                    "Success (Perf)": progress_perf[0],
                    "Success (Backup)": progress_backup[0],
                    "Success (Value)": progress_perf_shield_rej[0],
                    "cnt_safety_violation": cnt_safety_violation,
                    "cnt_num_episode": cnt_num_episode,
                    "buffer traj": len(self.memory),
                    "buffer sample": self.memory.num_sample,
                }, step=self.cnt_step, commit=True)

            # Saves model
            if save_metric == 'perf':
                self.save(metric=progress_perf[0])
            elif save_metric == 'backup':
                self.save(metric=progress_backup[0])
            else:
                raise NotImplementedError

            # Re-initializes pb to avoid memory explosion from mesh loading
            # - this will also terminates any trajectories at sampling
            venv.env_method('close_pb')
            s, _prev_obs, _prev_obs_stack = self.reset_all_envs(
                venv, random_init=random_init
            )

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
                    figure_folder_perf,
                    plot_backup=True,
                    figure_folder_backup=figure_folder_backup,
                    plot_shield=True,
                    figure_folder_shield=figure_folder_shield,
                    shield_dict=shield_dict,
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                )

            # Releases GPU RAM as much as possible
            torch.cuda.empty_cache()

            # Re-initializes env
            env.close_pb()
            env.reset()

        self.save(force_save=True)
        end_learning = time.time()
        time_learning = end_learning - start_learning
        print('\nLearning: {:.1f}'.format(time_learning))

        for i, tp in enumerate(train_records):
            train_records[i] = np.array(tp)
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        violation_record = np.array(violation_record)
        episode_record = np.array(episode_record)
        return train_records, train_progress, violation_record, episode_record

    def store_traj_online(self, env_ind):
        self.memory_online.update(self.trajs[env_ind])
        self.trajs[env_ind] = []

    def sample_batch_traj_online(
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
        trajs, trajs_nxt = self.memory_online.sample(
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

    def unpack_batch_backup(
        self, batch: Tuple, batch_nxt: Tuple, get_perf_action: bool,
        get_latent: bool = False, get_latent_backup: bool = False,
        no_stack: bool = False
    ):
        """Unpacks a batch of trajectories.

        Args:
            batch (tuple): list of trajectories.
            batch_nxt (tuple): list of trajectories.
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

        if traj_all[-1][-1]['a_'].dim() == 1:
            action_next = torch.FloatTensor([
                traj[-1]['a_'] for traj in traj_all
            ])
        else:
            action_next = torch.cat([traj[-1]['a_'] for traj in traj_all])
        action_next = action_next.to(self.device)
        action_next = action_next[non_final_mask]

        return (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x,
            latent, append, non_final_append_nxt, l_x_ra, binary_cost, hn, cn,
            action_next
        )
