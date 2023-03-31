# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training prior for PAC-Base and PAC-RP.

This file implements the training of the prior policy with latent-conditioned
performance policy without backup policy or shielding, which are the methods
PAC-Base and PAC-RP in the paper.

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

from agent.base_training import BaseTraining
from agent.SAC_maxEnt import SAC_maxEnt
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class PolicyPriorBasePAC(BaseTraining):

    def __init__(self, cfg, cfg_performance, cfg_env, verbose: bool = True):
        """
        Args:
            cfg: config for agent
            cfg_performance: config for performance policy
            cfg_backup: config for backup policy
            cfg_env: config for environment
            verbose: whether to print verbose info
        """
        super().__init__(cfg, cfg_env, cfg_performance.train)
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

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the prior of PAC-Base and PAC-RP.

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
        opt_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        disc_num_update_per_opt = self.cfg.disc_update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
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
        s, task_ids = venv.reset(random_init=False)
        z = self.performance.latent_prior.sample((self.n_envs,))
        z = z.to(self.device)

        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append for performance policy
            append = venv.get_append(venv.get_attr('_state'))

            # Selects action - random if before 1st optimize
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

            # Gets new append
            append_next = venv.get_append(venv.get_attr('_state'))
            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):

                # Saves append
                info['append'] = append[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next[env_ind].unsqueeze(0)

                info['a_perf'] = a_exec[env_ind, :-1].unsqueeze(0)
                # Stores the transition in memory. Although no_shield does not
                # have the backup agent, we follow the common store_transition
                # as in perf_latent.
                self.store_transition(
                    z[env_ind].clone().unsqueeze(0),
                    s[env_ind].unsqueeze(0).to(self.image_device),
                    a_exec[env_ind].unsqueeze(0), r,
                    s_.unsqueeze(0).to(self.image_device), done, info
                )

                # Checks if episode finished
                if done:

                    # Re-sample z
                    z[env_ind] = self.performance.latent_prior.sample().to(
                        self.device
                    )

                    # Resets environment with randomly sampled task
                    obs, task_id = venv.reset_one(env_ind, random_init=False)

                    # Updates task id
                    task_ids[env_ind] = task_id

                    # Updates observations
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
            if (
                self.cnt_step >= min_step_b4_opt and cnt_opt_period >= opt_freq
            ):
                cnt_opt_period = 0
                loss_perf = np.zeros(5)  # disc

                # Updates discriminator
                loss_disc = []
                for _ in range(disc_num_update_per_opt):
                    _online_batch = self.sample_batch(
                        batch_size=self.disc_batch_size,
                        recent_size=self.disc_recent_size
                    )
                    online_batch = self.unpack_batch(
                        _online_batch, get_latent=True, get_perf_action=True
                    )
                    loss_disc += [self.performance.update_disc(online_batch)]
                loss_disc = np.mean(loss_disc)

                # Updates critic/actor
                for timer in range(num_update_per_opt):
                    batch = self.sample_batch()
                    batch_perf = self.unpack_batch(
                        batch, get_latent=True, get_perf_action=True
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
                s, task_ids = venv.reset(random_init=False)

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
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
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
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
                        )
                    train_progress[0].append(progress_perf)
                    train_progress[1].append(self.cnt_step)
                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success": progress_perf[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
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
                            venv, env, plot_v, vmin, vmax, save_figure,
                            plot_figure, figure_folder, plot_backup=False,
                            figure_folder_backup=None, plot_shield=False,
                            figure_folder_shield=None, shield_dict=None
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
