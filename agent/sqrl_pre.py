# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training prior for SQRL.

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

from agent.base_training import BaseTraining, TransitionLatent
from agent.replay_memory import ReplayMemory
from agent.SAC_mini import SAC_mini
from agent.SAC_sqrl import SAC_sqrl
from utils.misc import check_shielding
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class SQRLPre(BaseTraining):

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
        super().__init__(cfg, cfg_env, cfg_performance.train)

        print("= Constructing performance agent")
        self.performance = SAC_mini(
            cfg_performance.train, cfg_performance.arch, cfg_env
        )
        self.performance.build_network(verbose=verbose)

        print("= Constructing backup agent")
        self.backup = SAC_sqrl(cfg_backup.train, cfg_backup.arch, cfg_env)
        self.backup.build_network(verbose=verbose)

        # the first stage in SQRL needs to have two separate buffers
        self.memory_online = ReplayMemory(cfg.on_memory_capacity, cfg.seed)

        # For saving and removing models
        self.module_all = [self.performance, self.backup]

    @property
    def has_backup(self):
        return True

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the prior of SQRL.

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

        # Shielding info
        shield_dict = self.cfg.train_shield_dict

        z = None
        while self.cnt_step < max_steps:
            # Use performance policy for the first part of the cycle
            s, _ = venv.reset(random_init=False, state_init=None)
            cnt_opt_period = 0
            print(self.cnt_step, end='\r')

            while cnt_perf_steps < perf_steps:
                # Saves robot state for reloading
                _prev_states = np.array(venv.get_attr('_state'))

                # Setss train modes for all envs
                venv.env_method('set_train_mode')

                # Gets append
                append = venv.get_append(_prev_states)

                # Selects action
                with torch.no_grad():
                    a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                    a_exec = a_exec.to(self.device)
                    a_perf, _ = self.performance.actor.sample(
                        s, append=append, latent=z
                    )
                    a_exec[:, :-1] = a_perf
                    a_exec[:, -1] = self.performance.act_ind

                # Applies action - update heading
                s_all, r_all, done_all, info_all = venv.step(a_exec)
                append_next = venv.get_append(venv.get_attr('_state'))
                for env_ind, (s_, r, done, info) in enumerate(
                    zip(s_all, r_all, done_all, info_all)
                ):
                    # Saves append
                    info['append'] = append[env_ind].unsqueeze(0)
                    info['append_nxt'] = append_next[env_ind].unsqueeze(0)
                    info['a_perf'] = a_exec[env_ind, :-1].unsqueeze(0)

                    # Stores the transition in memory
                    self.store_transition(
                        z, s[env_ind].unsqueeze(0).to(self.image_device),
                        a_exec[env_ind].unsqueeze(0), r,
                        s_.unsqueeze(0).to(self.image_device), done, info
                    )

                    # Resets if failure or timeout
                    if done:
                        obs, _ = venv.reset_one(env_ind)
                        s_all[env_ind] = obs
                        g_x = info['g_x']
                        if g_x > 0:
                            cnt_safety_violation += 1
                        cnt_num_episode += 1
                violation_record.append(cnt_safety_violation)
                episode_record.append(cnt_num_episode)

                # Updatess "prev" states
                s = s_all

                # Counts
                self.cnt_step += self.n_envs
                cnt_opt_period += self.n_envs
                if cnt_perf_steps < perf_steps:
                    cnt_perf_steps += self.n_envs

                # Optimizes
                if (cnt_opt_period >= opt_freq):
                    print("Performance optimization starts")
                    cnt_opt_period = 0
                    loss_perf = np.zeros(4)

                    for timer in range(num_update_per_opt):
                        # update performance using offline replay buffer
                        if len(self.memory) >= min_step_b4_opt:
                            batch_perf = self.unpack_batch(
                                self.sample_batch(), get_latent=False,
                                get_perf_action=True
                            )
                            loss_tp_perf = self.performance.update(
                                batch_perf, timer,
                                update_period=self.update_period
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

            # Use rejection sampling for the second part of the cycle
            init_flag_all = torch.full(size=(self.n_envs,), fill_value=True)
            s_old = None
            a_old = None
            done_old = None
            r_old = None
            info_old = None
            append_old = None
            s, _ = venv.reset(random_init=False, state_init=None)
            cnt_opt_period = 0
            print("On-Policy Samples")
            while cnt_backup_steps < backup_steps:
                # Saves robot state for reloading
                _prev_states = np.array(venv.get_attr('_state'))

                # Setss train modes for all envs
                venv.env_method('set_train_mode')

                # Gets append
                append = venv.get_append(_prev_states)

                # Selects action
                with torch.no_grad():
                    a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                    a_exec = a_exec.to(self.device)
                    a_perf, _ = self.performance.actor.sample(
                        s, append=append, latent=z
                    )
                    a_exec[:, :-1] = a_perf
                    a_exec[:, -1] = self.performance.act_ind
                a_shield = a_exec.clone()

                # Shielding
                _, shield_info = check_shielding(
                    self.backup, shield_dict, s, a_shield, append,
                    context_backup=None, state=_prev_states,
                    policy=self.performance, context_policy=z
                )
                a_shield = shield_info['action_final']

                # Applies action - update heading
                s_all, r_all, done_all, info_all = venv.step(a_shield)
                for env_ind, (done, init_flag) in enumerate(
                    zip(done_all, init_flag_all)
                ):
                    # Stores the transition in memory
                    if not init_flag:
                        info = deepcopy(info_old[env_ind])
                        info['a_'] = a_shield[env_ind].unsqueeze(0)
                        info['append'] = append_old[env_ind].unsqueeze(0)
                        info['append_nxt'] = append[env_ind].unsqueeze(0)
                        self.store_transition_online(
                            z,
                            s_old[env_ind].unsqueeze(0).to(self.image_device),
                            a_old[env_ind].unsqueeze(0), r_old[env_ind],
                            s[env_ind].unsqueeze(0).to(self.image_device),
                            done_old[env_ind], info
                        )

                    # Resets if failure or timeout
                    if done:
                        obs, _ = venv.reset_one(env_ind)
                        s_all[env_ind] = obs
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
                s = s_all

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
                    for timer in range(num_update_per_opt):
                        # update backup using online replay buffer
                        if len(self.memory_online) >= min_step_b4_opt:
                            batch_backup = self.unpack_batch(
                                self.sample_batch(online=True),
                                get_latent=False, get_perf_action=False,
                                get_action_next=True
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

            # Resets counter of steps
            cnt_perf_steps = 0
            cnt_backup_steps = 0

            # Checks after a cycle
            print(
                '\n  - Safety violations so far: {:d}'.
                format(cnt_safety_violation)
            )

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
                )
                progress_backup = self.backup.check(
                    venv,
                    self.cnt_step,
                    states=sample_states,
                    check_type='all_env',
                    verbose=True,
                    num_traj_per_env=num_traj_per_env,
                    revert_task=True,
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
                    shield_dict=self.cfg.train_shield_dict,
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
                    shield_dict=self.cfg.train_shield_dict,
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
            s, _ = venv.reset()

            # Saves training details
            torch.save({
                'train_records': train_records,
                'train_progress': train_progress,
                'violation_record': violation_record,
                "episode_record": episode_record,
            }, os.path.join(out_folder, 'train_details'))

            if plot_figure or save_figure:
                self.get_figures(
                    venv, env, plot_v, vmin, vmax, save_figure, plot_figure,
                    figure_folder_perf, plot_backup=True,
                    figure_folder_backup=figure_folder_backup,
                    plot_shield=True,
                    figure_folder_shield=figure_folder_shield,
                    shield_dict=shield_dict
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

    def store_transition_online(self, *args):
        """
        Stores transitions into the online replay buffer. It differs in the
        base_training class because here the online buffer is smaller in size
        and stores different actions.
        """
        self.memory_online.update(TransitionLatent(*args))

    def sample_batch(
        self, batch_size: Optional[int] = None, online: bool = False
    ):
        """
        Samples from the replay buffer. It differs in the base_training class
        because the online buffer is different.

        Args:
            batch_size (int, optional): the size of the batch used for update.
                Defaults to None.
            online (bool, optional): sample from the online buffer if True.
                Defaults to False.

        Returns:
            Transition of batches.
        """
        if online:
            sample = self.memory_online.sample
        else:
            sample = self.memory.sample
        if batch_size is None:
            transitions = sample(self.batch_size)[0]
        else:
            transitions = sample(batch_size)[0]
        batch = TransitionLatent(*zip(*transitions))
        return batch

    def unpack_batch(
        self, batch: Tuple, get_latent: bool = False,
        get_perf_action: bool = False, get_l_x_ra: bool = False,
        get_action_next: bool = False
    ):
        _batch = super().unpack_batch(
            batch, get_latent, get_perf_action, get_l_x_ra
        )

        if get_action_next:
            if batch.info[0]['a_'].dim() == 1:
                action_next = torch.FloatTensor([
                    info['a_'] for info in batch.info
                ])
            else:
                action_next = torch.cat([info['a_'] for info in batch.info])
            action_next = action_next.to(self.device)

            return _batch + (action_next,)
        else:
            return _batch
