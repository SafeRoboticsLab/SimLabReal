# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training posterior of PAC-Shield-Perf.

This file implements the training of the posterior policy with
latent-conditioned performance policy and backup policy without conditioning on
latent variable, which is the method PAC-Shield-Perf in the paper.

In the posterior stage, only the latent distribution is fine-tuned. The loss
function includes the task rewards and the PAC-Bayes bound regularization
between the prior and posterior. See SAC_ps.py for the details.
"""

from typing import Optional, Tuple
import torch
import numpy as np
import os
import time
import wandb
from shutil import copyfile

from agent.base_training import BaseTraining
from agent.SAC_mini import SAC_mini
from agent.SAC_ps_c import SAC_ps_c
from utils.misc import get_kl_bound, get_renyi_bound, check_shielding
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class PolicyPosteriorPerfLatent(BaseTraining):

    def __init__(
        self, cfg, cfg_performance, cfg_backup, cfg_env, verbose: bool = False
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
        #! This bool parameter controls whether using action relabeling. We
        #! argue that this parameter should always be True. We add the option
        #! here for testing the difference.
        self.action_relabel = cfg.action_relabel

        self.cfg_performance_train = cfg_performance.train
        self.cfg_backup_train = cfg_backup.train
        self.N = cfg_env.num_env_train
        self.delta = cfg.delta

        print("= Constructing performance agent")
        self.performance = SAC_ps_c(
            cfg_performance.train, cfg_performance.arch, cfg_env
        )
        self.performance.build_network(
            verbose=verbose, actor_path=cfg_performance.train.actor_path,
            critic_path=cfg_performance.train.critic_path
        )

        print("= Constructing backup agent")
        self.backup = SAC_mini(cfg_backup.train, cfg_backup.arch, cfg_env)
        self.backup.build_network(
            verbose=verbose, actor_path=cfg_backup.train.actor_path,
            critic_path=cfg_backup.train.critic_path, build_optimizer=False
        )

        # For saving and removing models
        self.module_all = [self.performance]

    @property
    def has_backup(self):
        return True

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the posterior of PAC-Shield-Perf.

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
        min_step_before_opt = self.cfg.min_steps_b4_opt
        num_traj_per_env = self.cfg.num_traj_per_env
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder

        # == Main Training ==
        start_learning = time.time()
        train_records = []
        train_progress = [[], [], []]  # perf, shield, kl/renyi
        violation_record = []
        episode_record = []
        latent_record = [[], []]  # mean, std
        cnt_opt = 0
        cnt_opt_period = 0
        cnt_safety_violation = 0
        cnt_num_episode = 0

        # Model folders
        model_folder = os.path.join(out_folder, 'model')
        model_folder_perf = os.path.join(model_folder, 'performance')
        model_folder_backup = os.path.join(model_folder, 'backup')
        os.makedirs(model_folder_perf, exist_ok=True)
        os.makedirs(model_folder_backup, exist_ok=True)
        self.module_folder_all = [model_folder_perf]
        save_metric = self.cfg.save_metric

        # Saves a copy of models
        copyfile(
            self.cfg_performance_train.actor_path,
            os.path.join(model_folder_perf, 'init_actor.pth')
        )
        if self.cfg_performance_train.critic_path is not None:
            copyfile(
                self.cfg_performance_train.critic_path,
                os.path.join(model_folder_perf, 'init_critic.pth')
            )
        copyfile(
            self.cfg_backup_train.actor_path,
            os.path.join(model_folder_backup, 'init_actor.pth')
        )
        copyfile(
            self.cfg_backup_train.critic_path,
            os.path.join(model_folder_backup, 'init_critic.pth')
        )

        if save_figure:
            figure_folder = os.path.join(out_folder, 'figure')
            figure_folder_perf = os.path.join(figure_folder, 'performance')
            figure_folder_shield = os.path.join(figure_folder, 'shield')
            os.makedirs(figure_folder_perf, exist_ok=True)
            os.makedirs(figure_folder_shield, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Evaluate prior
        env.set_eval_mode()  # always eval mode
        venv.env_method('set_eval_mode')
        sample_states = None
        progress_perf = self.performance.check(
            env=venv,
            cnt_step=self.cnt_step,
            states=sample_states,
            check_type='all_env',
            verbose=True,
            num_traj_per_env=num_traj_per_env,
            revert_task=True,
            # kwargs
            latent_dist=self.performance.latent_prior,
            fixed_init=True
        )
        progress_perf_shield_value = self.performance.check(
            env=venv,
            cnt_step=self.cnt_step,
            states=sample_states,
            check_type='all_env',
            verbose=True,
            num_traj_per_env=num_traj_per_env,
            revert_task=True,
            # kwargs
            latent_dist=self.performance.latent_prior,
            shield=True,
            backup=self.backup,
            shield_dict=shield_dict,
            fixed_init=True
        )
        train_progress[0].append(progress_perf)
        train_progress[1].append(progress_perf_shield_value)
        if self.cfg.use_wandb:
            wandb.log({
                "Success (Perf)": progress_perf[0],
                "Success (Value)": progress_perf_shield_value[0]
            }, step=self.cnt_step, commit=True)

        if plot_figure or save_figure:
            self.get_figures(
                venv, env, plot_v, vmin, vmax, save_figure, plot_figure,
                figure_folder_perf, plot_backup=False,
                figure_folder_backup=None, plot_shield=True,
                plot_shield_value=True,
                figure_folder_shield=figure_folder_shield,
                shield_dict=shield_dict
            )

        # Resets all envs
        s, _ = venv.reset(random_init=False, state_init=None)
        z = self.performance.latent_ps.sample((self.n_envs,)).to(self.device)
        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append
            _states = np.array(venv.get_attr('_state'))
            append = venv.get_append(_states)

            # Selects action: the stochasticity in the posterior training only
            # comes from z.
            with torch.no_grad():
                a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                a_exec = a_exec.to(self.device)
                a_perf = self.performance.actor(
                    s, append=append, latent=z
                )  # condition on latent
                a_exec[:, :-1] = a_perf
                a_exec[:, -1] = self.performance.act_ind

            # Make a copy for proposed actions
            a_shield = a_exec.clone()

            # Always shield using value
            shield_flag, _ = check_shielding(
                self.backup, shield_dict, s, a_exec, append,
                context_backup=None, state=_states, policy=self.performance,
                context_policy=z
            )
            if torch.any(shield_flag):
                a_backup = self.backup.actor(
                    s[shield_flag], append=append[shield_flag]
                ).data
                a_shield[shield_flag, :-1] = a_backup
                a_shield[shield_flag, -1] = self.backup.act_ind

            # Interact with env
            s_all, r_all, done_all, info_all = venv.step(a_shield)
            append_next = venv.get_append(venv.get_attr('_state'))
            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):
                # Saves extra info: a_perf and append
                info['a_perf'] = a_perf[env_ind].unsqueeze(0)
                info['append'] = append[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next[env_ind].unsqueeze(0)

                # Stores the transition in memory
                self.store_transition(
                    z[env_ind].clone().unsqueeze(0),
                    s[env_ind].unsqueeze(0).to(self.image_device),
                    a_shield[env_ind].unsqueeze(0), r,
                    s_.unsqueeze(0).to(self.image_device), done, info
                )

                # Checks if episode finished
                if done:
                    obs, _ = venv.reset_one(env_ind, random_init=False)
                    s_all[env_ind] = obs
                    z[env_ind] = (
                        self.performance.latent_ps.sample().to(self.device)
                    )

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
            if (
                self.cnt_step >= min_step_before_opt
                and cnt_opt_period >= opt_freq
            ):
                cnt_opt_period = 0

                # Do not update actor (latent distribution) in 1st update -
                # let critic update a bit first since reward function changed
                # without maxEnt
                if self.cnt_step == min_step_before_opt:
                    update_period = num_update_per_opt
                else:
                    update_period = self.update_period

                # Updates critic/actor
                loss_perf = np.zeros(3)
                for timer in range(num_update_per_opt):
                    batch = self.unpack_batch(
                        self.sample_batch(), get_latent=True,
                        get_perf_action=self.action_relabel
                    )
                    loss_tp_perf = self.performance.update(
                        batch, timer, update_period=update_period
                    )
                    for i, l in enumerate(loss_tp_perf):
                        loss_perf[i] += l

                # Average for losses
                loss_perf /= num_update_per_opt

                # Record
                train_records.append(loss_perf)
                latent_mean = (
                    self.performance.latent_mean.clone().cpu().numpy()
                )
                latent_std = self.performance.latent_std.clone().cpu().numpy()
                latent_record[0].append(latent_mean)
                latent_record[1].append(latent_std)
                if self.cfg.use_wandb:
                    for idx, (z_m, z_std) in enumerate(
                        zip(latent_mean, latent_std)
                    ):
                        wandb.log({
                            "z_mean_" + str(idx): z_m,
                            "z_std_" + str(idx): z_std,
                        }, step=self.cnt_step, commit=False)
                    wandb.log({
                        "loss_q (Perf)": loss_perf[0],
                        "loss_pi (Perf)": loss_perf[1],
                        "loss_bound_reg (Perf)": loss_perf[2],
                    }, step=self.cnt_step, commit=False)

                # Re-initializes pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, _ = venv.reset(random_init=False)

                # Counts number of optimization
                cnt_opt += 1

                # Checks after fixed number of steps
                if cnt_opt % check_opt_freq == 0:
                    print(
                        '\n  - Safety violations so far: {:d}'.
                        format(cnt_safety_violation)
                    )

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    venv.env_method('set_eval_mode')

                    # Sets states for check()
                    progress_perf = self.performance.check(
                        env=venv,
                        cnt_step=self.cnt_step,
                        states=sample_states,
                        check_type='all_env',
                        verbose=True,
                        num_traj_per_env=num_traj_per_env,
                        revert_task=True,
                        # kwargs
                        latent_dist=self.performance.latent_ps,
                        fixed_init=True
                    )
                    progress_perf_shield_value = self.performance.check(
                        env=venv,
                        cnt_step=self.cnt_step,
                        states=sample_states,
                        check_type='all_env',
                        verbose=True,
                        num_traj_per_env=num_traj_per_env,
                        revert_task=True,
                        # kwargs
                        latent_dist=self.performance.latent_ps,
                        shield=True,
                        backup=self.backup,
                        shield_dict=shield_dict,
                        fixed_init=True
                    )

                    # Bound - assume check and update at the same time
                    with torch.no_grad():
                        kl_div = self.performance.get_kl_div().cpu().numpy()
                        renyi_div = self.performance.get_renyi_div().cpu(
                        ).numpy()
                    kl_bound = get_kl_bound(
                        progress_perf_shield_value[0], kl_div, self.N,
                        self.delta
                    )
                    renyi_bound = get_renyi_bound(
                        progress_perf_shield_value[0], renyi_div, self.N,
                        self.delta
                    )

                    # Log
                    train_progress[0].append(progress_perf)
                    train_progress[1].append(progress_perf_shield_value)
                    train_progress[2].append([
                        self.cnt_step, kl_bound, renyi_bound
                    ])

                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success (Perf)": progress_perf[0],
                            "Success (Value)": progress_perf_shield_value[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
                            "KL bound": kl_bound,
                            "Renyi bound": renyi_bound,
                            "KL div": kl_div,
                            "Renyi div": renyi_div
                        }, step=self.cnt_step, commit=True)

                    # Saves model
                    if save_metric == 'value':
                        self.save(metric=progress_perf_shield_value[0])
                    elif save_metric == 'kl_bound':
                        self.save(metric=kl_bound)
                    elif save_metric == 'renyi_bound':
                        self.save(metric=renyi_bound)
                    else:
                        raise NotImplementedError

                    # Saves training details
                    torch.save({
                        'train_records': train_records,
                        'train_progress': train_progress,
                        'violation_record': violation_record,
                        "episode_record": episode_record,
                        'latent_record': latent_record
                    }, os.path.join(out_folder, 'train_details'))

                    # Plot
                    if plot_figure or save_figure:
                        self.get_figures(
                            venv, env, plot_v, vmin, vmax, save_figure,
                            plot_figure, figure_folder_perf, plot_backup=False,
                            figure_folder_backup=None, plot_shield=True,
                            plot_shield_value=True,
                            figure_folder_shield=figure_folder_shield,
                            shield_dict=shield_dict
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
