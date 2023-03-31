# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""A class for training Recovery RL.

This file implements the training of a performance policy and a backup policy,
which is the method Recovery RL in the paper.

The performance policy is trained with task rewards. The backup policy is
trained with binary collision indicators (sparse rewards).

Paper: https://ieeexplore.ieee.org/document/9392290.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import torch
import wandb
from shutil import copyfile

from .SAC_mini import SAC_mini
from .base_training import BaseTraining
from utils.misc import check_shielding
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class RecoveryRL(BaseTraining):
    def __init__(self,
                 cfg,
                 cfg_performance,
                 cfg_backup,
                 cfg_env,
                 verbose: bool = False):
        """
        Args:
            cfg: config for agent
            cfg_performance: config for performance policy
            cfg_backup: config for backup policy
            cfg_env: config for environment
            verbose: whether to print verbose info
        """
        super().__init__(cfg, cfg_env, cfg_performance.train)
        self.cfg_backup_train = cfg_backup.train

        print("= Constructing performance agent")
        self.performance = SAC_mini(cfg_performance.train,
                                    cfg_performance.arch, cfg_env)
        self.performance.build_network(verbose=verbose)

        print("= Constructing backup agent")
        self.backup = SAC_mini(cfg_backup.train, cfg_backup.arch, cfg_env)

        if cfg_backup.train.pre_train:
            # restore pretrained model if specified
            self.backup.build_network(verbose=verbose,
                                      actor_path=cfg_backup.train.actor_path,
                                      critic_path=cfg_backup.train.critic_path)
        else:
            self.backup.build_network(verbose=verbose)

        # if true, train backup also
        self.train_backup = cfg_backup.train.train_backup

        # For saving and removing models
        if self.train_backup:
            self.module_all = [self.performance, self.backup]
        else:
            self.module_all = [self.performance]

    @property
    def has_backup(self):
        return True

    def learn(
        self,
        venv: VecEnvWrapper,
        env: BaseEnv,
        current_step: Optional[int] = None,
        vmin: float = -1,
        vmax: float = 1,
        save_figure: bool = True,
        plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the Recovery RL.

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
        shield_dict = self.cfg.shield_dict
        max_steps = self.cfg.max_steps
        opt_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
        min_step_b4_opt = self.cfg.min_steps_b4_opt
        num_traj_per_env = self.cfg.num_traj_per_env
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder
        sample_states = None

        # == Main Training ==
        start_learning = time.time()
        train_records = [[]]
        train_progress = [[], [], []]  # performance, value-shielded
        violation_record = []
        episode_record = []
        cnt_opt = 0
        cnt_opt_period = 0
        cnt_safety_violation = 0
        cnt_num_episode = 0

        # Saves model
        model_folder = os.path.join(out_folder, 'model')
        model_folder_perf = os.path.join(model_folder, 'performance')
        model_folder_backup = os.path.join(model_folder, 'backup')
        os.makedirs(model_folder_perf, exist_ok=True)
        os.makedirs(model_folder_backup, exist_ok=True)
        save_metric = self.cfg.save_metric

        # Saves a copy of models
        copyfile(self.cfg_backup_train.actor_path,
                 os.path.join(model_folder_backup, 'init_actor.pth'))
        copyfile(self.cfg_backup_train.critic_path,
                 os.path.join(model_folder_backup, 'init_critic.pth'))

        if save_figure:
            figure_folder = os.path.join(out_folder, 'figure')
            figure_folder_perf = os.path.join(figure_folder, 'performance')
            figure_folder_shield = os.path.join(figure_folder, 'shield')
            os.makedirs(figure_folder_perf, exist_ok=True)
            os.makedirs(figure_folder_shield, exist_ok=True)

        # train backup
        figure_folder_backup = None
        if self.train_backup:
            if save_figure:
                figure_folder_backup = os.path.join(figure_folder, 'backup')
                os.makedirs(figure_folder_backup, exist_ok=True)
            train_records = [[], []]
            train_progress = [[], [], [], []]  # add backup
            self.module_folder_all = [model_folder_perf, model_folder_backup]
        else:
            self.module_folder_all = [model_folder_perf]

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # prior
        progress_perf = self.performance.check(
            venv,
            self.cnt_step,
            states=sample_states,
            check_type='all_env',
            verbose=True,
            num_traj_per_env=num_traj_per_env,
            revert_task=True,
            # kwargs for env.simulate_trajectories
            latent_dist=None,
            fixed_init=True)
        progress_perf_shield_value = self.performance.check(
            venv,
            self.cnt_step,
            states=sample_states,
            check_type='all_env',
            verbose=True,
            num_traj_per_env=num_traj_per_env,
            revert_task=True,
            # kwargs for env.simulate_trajectories
            latent_dist=None,
            shield=True,
            backup=self.backup,
            shield_dict=shield_dict,
            fixed_init=True)
        train_progress[0].append(progress_perf)
        train_progress[1].append(progress_perf_shield_value)
        if self.train_backup:
            progress_backup = self.backup.check(
                venv,
                self.cnt_step,
                states=sample_states,
                check_type='all_env',
                verbose=True,
                num_traj_per_env=num_traj_per_env,
                revert_task=True,
                # kwargs for env.simulate_trajectories
                latent_dist=None,
                fixed_init=True)
            train_progress[2].append(progress_backup)
        train_progress[-1].append(self.cnt_step)
        if self.cfg.use_wandb:
            wandb.log(
                {
                    "Success (Perf)": progress_perf[0],
                    "Success (Value)": progress_perf_shield_value[0],
                },
                step=self.cnt_step,
                commit=not self.train_backup)
            if self.train_backup:
                wandb.log({
                    "Success (Backup)": progress_backup[0],
                },
                          step=self.cnt_step,
                          commit=True)

        # Resets all envs
        s, _ = venv.reset(random_init=False, state_init=None)
        z = None
        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append
            _states = np.array(venv.get_attr('_state'))
            append = venv.get_append(_states)

            # Selects action
            with torch.no_grad():
                a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                a_exec = a_exec.to(self.device)
                a_perf, _ = self.performance.actor.sample(s,
                                                          append=append,
                                                          latent=None)
                a_exec[:, :-1] = a_perf
                a_exec[:, -1] = self.performance.act_ind

            # Make a copy for proposed actions
            a_shield = a_exec.clone()

            # Always shield using value
            shield_flag, _ = check_shielding(self.backup,
                                             shield_dict,
                                             s,
                                             a_exec,
                                             append,
                                             context_backup=None,
                                             state=_states,
                                             policy=self.performance,
                                             context_policy=z)
            if torch.any(shield_flag):
                a_backup, _ = self.backup.actor.sample(
                    s[shield_flag], append=append[shield_flag], latent=None)
                a_shield[shield_flag, :-1] = a_backup.data
                a_shield[shield_flag, -1] = self.backup.act_ind

            # Interact with env
            s_all, r_all, done_all, info_all = venv.step(a_shield)
            append_next = venv.get_append(venv.get_attr('_state'))
            for env_ind, (s_, r, done, info) in enumerate(
                    zip(s_all, r_all, done_all, info_all)):

                # Saves extra info: a_perf and append
                info['a_perf'] = a_perf[env_ind].unsqueeze(0)
                info['append'] = append[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next[env_ind].unsqueeze(0)

                # Stores the transition in memory
                self.store_transition(
                    None, s[env_ind].unsqueeze(0).to(self.image_device),
                    a_shield[env_ind].unsqueeze(0), r,
                    s_.unsqueeze(0).to(self.image_device), done, info)

                # Resample z
                if done:
                    obs, _ = venv.reset_one(env_ind, random_init=False)
                    s_all[env_ind] = obs

                    # Checks safety violation
                    g_x = info['g_x']
                    if g_x > 0:
                        cnt_safety_violation += 1
                    cnt_num_episode += 1
            violation_record.append(cnt_safety_violation)
            episode_record.append(cnt_num_episode)

            # Updatess "prev" states
            s = s_all

            # Optimizes
            if (self.cnt_step >= min_step_b4_opt
                    and cnt_opt_period >= opt_freq):
                cnt_opt_period = 0
                loss_perf = np.zeros(4)
                if self.train_backup:
                    loss_backup = np.zeros(4)

                # Updates critic/actor
                for timer in range(num_update_per_opt):
                    batch = self.sample_batch()
                    batch_perf = self.unpack_batch(batch,
                                                   get_latent=False,
                                                   get_perf_action=True)

                    loss_tp_perf = self.performance.update(
                        batch_perf, timer, self.update_period)
                    for i, l in enumerate(loss_tp_perf):
                        loss_perf[i] += l

                    if self.train_backup:
                        batch_backup = self.unpack_batch(batch,
                                                         get_latent=False,
                                                         get_perf_action=False)
                        loss_tp_backup = self.backup.update(
                            batch_backup, timer, self.update_period)
                        for i, l in enumerate(loss_tp_backup):
                            loss_backup[i] += l

                loss_perf /= num_update_per_opt
                if self.train_backup:
                    loss_backup /= num_update_per_opt

                # Record
                train_records[0].append(loss_perf)
                if self.train_backup:
                    train_records[1].append(loss_backup)
                if self.cfg.use_wandb:
                    lr = self.performance.critic_optimizer.state_dict(
                    )['param_groups'][0]['lr']
                    wandb.log(
                        {
                            "loss_q (Perf)": loss_perf[0],
                            "loss_pi (Perf)": loss_perf[1],
                            "loss_entropy (Perf)": loss_perf[2],
                            "loss_alpha (Perf)": loss_perf[3],
                            "learning rate (Perf)": lr
                        },
                        step=self.cnt_step,
                        commit=False)
                    if self.train_backup:
                        wandb.log(
                            {
                                "loss_q (Backup)": loss_backup[0],
                                "loss_pi (Backup)": loss_backup[1],
                                "loss_entropy (Backup)": loss_backup[2],
                                "loss_alpha (Backup)": loss_backup[3],
                            },
                            step=self.cnt_step,
                            commit=False)

                # Re-initializes pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, _ = venv.reset(random_init=False)

                # Counts number of optimization
                cnt_opt += 1

                # Checks after fixed number of steps
                if cnt_opt % check_opt_freq == 0:
                    print('\n  - Safety violations so far: {:d}'.format(
                        cnt_safety_violation))

                    # Sets states for check()
                    progress_perf = self.performance.check(
                        venv,
                        self.cnt_step,
                        states=sample_states,
                        check_type='all_env',
                        verbose=True,
                        num_traj_per_env=num_traj_per_env,
                        revert_task=True,
                        # kwargs for env.simulate_trajectories
                        latent_dist=None,
                        fixed_init=True)
                    progress_perf_shield_value = self.performance.check(
                        venv,
                        self.cnt_step,
                        states=sample_states,
                        check_type='all_env',
                        verbose=True,
                        num_traj_per_env=num_traj_per_env,
                        revert_task=True,
                        # kwargs for env.simulate_trajectories
                        latent_dist=None,
                        shield=True,
                        backup=self.backup,
                        shield_dict=shield_dict,
                        fixed_init=True)
                    train_progress[0].append(progress_perf)
                    train_progress[1].append(progress_perf_shield_value)
                    if self.train_backup:
                        progress_backup = self.backup.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='all_env',
                            verbose=True,
                            num_traj_per_env=num_traj_per_env,
                            revert_task=True,
                            # kwargs for env.simulate_trajectories
                            latent_dist=None,
                            fixed_init=True)
                        train_progress[2].append(progress_backup)
                    train_progress[-1].append(self.cnt_step)

                    if self.cfg.use_wandb:
                        wandb.log(
                            {
                                "Success (Perf)": progress_perf[0],
                                "Success (Value)":
                                progress_perf_shield_value[0],
                                "cnt_safety_violation": cnt_safety_violation,
                                "cnt_num_episode": cnt_num_episode,
                            },
                            step=self.cnt_step,
                            commit=not self.train_backup)
                        if self.train_backup:
                            wandb.log(
                                {
                                    "Success (Backup)": progress_backup[0],
                                },
                                step=self.cnt_step,
                                commit=True)

                    # Saves model
                    if self.train_backup and save_metric == 'backup':
                        self.save(metric=progress_backup[0])
                    elif save_metric == 'perf':
                        self.save(metric=progress_perf[0])
                    elif save_metric == 'value':
                        self.save(metric=progress_perf_shield_value[0])
                    else:
                        raise NotImplementedError

                    # Saves training details
                    torch.save(
                        {
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
                            plot_backup=self.train_backup,
                            figure_folder_backup=figure_folder_backup,
                            plot_shield=True,
                            figure_folder_shield=figure_folder_shield,
                            shield_dict=shield_dict)

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
                self.backup.update_hyper_param()

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
