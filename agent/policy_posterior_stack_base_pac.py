# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training posterior for PAC-Base and PAC-RP with frame stacking.

This file implements the training of the posterior policy with
latent-conditioned performance policy without backup policy or shielding,
which is the method PAC-Base and PAC-RP in the paper. Additionally,
the performance policy has stacked frames as input.

In the posterior stage, only the latent distribution is fine-tuned. The loss
function includes the task rewards (or safety) and the PAC-Bayes bound
regularization between the prior and posterior. See SAC_ps.py for the details.
"""

from typing import Optional, Tuple
import torch
import numpy as np
import os
import time
import wandb
from shutil import copyfile
from collections import deque

from agent.base_training_stack import BaseTrainingStack
from agent.SAC_ps_c import SAC_ps_c
from utils.misc import get_kl_bound, get_renyi_bound, get_frames
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class PolicyPosteriorStackBasePAC(BaseTrainingStack):

    def __init__(self, cfg, cfg_performance, cfg_env, verbose: bool = False):
        """
        Args:
            cfg: config for agent
            cfg_performance: config for performance policy
            cfg_backup: config for backup policy
            cfg_env: config for environment
            verbose: whether to print verbose info
        """
        super().__init__(cfg, cfg_env, cfg_performance.train, sample_next=True)
        self.cfg_performance_train = cfg_performance.train
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
        """Learns the posterior of PAC-Base and PAC-RP with frame stacking.

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
        optimize_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
        min_step_before_opt = self.cfg.min_steps_b4_opt
        num_traj_per_env = self.cfg.num_traj_per_env
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder
        sample_states = None
        # num_rnd_traj = num_traj_per_env * venv.num_task

        # == Main Training ==
        start_learning = time.time()
        train_records = []
        train_progress = [[], []]
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
        os.makedirs(model_folder_perf, exist_ok=True)
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

        if save_figure:
            figure_folder = os.path.join(out_folder, 'figure')
            os.makedirs(figure_folder, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Evaluate prior
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
            fixed_init=True,
            traj_size=self.traj_size,
            frame_skip=self.frame_skip,
        )
        train_progress[0].append(progress_perf)
        if self.cfg.use_wandb:
            wandb.log({
                "Success": progress_perf[0],
            }, step=self.cnt_step, commit=True)

        # Resets all envs
        s, _prev_obs, _prev_obs_stack = self.reset_all_envs(venv)
        z = self.performance.latent_ps.sample((self.n_envs,)).to(self.device)
        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Gets append
            _states = np.array(venv.get_attr('_state'))
            append = venv.get_append(_states)
            if self.use_append_noise:
                append_noise = self.append_noise_dist.sample((self.n_envs,))
                append += append_noise

            # Selects action: the stochasticity in the posterior training only
            # comes from z.
            with torch.no_grad():
                a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                a_exec = a_exec.to(self.device)
                a_perf = self.performance.actor(
                    _prev_obs_stack, append=append, latent=z
                )  # condition on latent
                a_exec[:, :-1] = a_perf
                a_exec[:, -1] = self.performance.act_ind

            # Make a copy for proposed actions
            a_shield = a_exec.clone()

            # Interact with env
            s_all, r_all, done_all, info_all = venv.step(a_shield)

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
                # Saves extra info: a_perf and append
                info['a_perf'] = a_perf[env_ind].unsqueeze(0)
                info['append'] = append[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next[env_ind].unsqueeze(0)

                # Stores the transition in memory
                self.store_transition_traj(
                    env_ind, z[env_ind].clone().unsqueeze(0),
                    s[env_ind].unsqueeze(0).to(self.image_device),
                    a_shield[env_ind].unsqueeze(0), r, done, info
                )

                # Resets if failure or timeout
                if done:
                    # Pushes traj to replay buffer and reset traj buffer for
                    # that env.
                    self.store_traj(env_ind)

                    # Resets environment with randomly sampled task
                    obs, _ = venv.reset_one(env_ind, verbose=False)
                    _prev_obs[env_ind] = deque(maxlen=self.traj_cover)

                    # Updates observations
                    s_all[env_ind] = obs
                    _prev_obs[env_ind].appendleft(obs)
                    z[env_ind] = self.performance.latent_ps.sample().to(
                        self.device
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
            _prev_obs_stack = _obs_stack

            # Optimizes
            if (
                self.cnt_step >= min_step_before_opt
                and cnt_opt_period >= optimize_freq
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
                loss_perf = [[], [], [], []]
                self.memory.set_possible_samples(
                    traj_size=self.traj_size,
                    frame_skip=self.frame_skip,
                    allow_repeat_frame=False,
                    recent_size=None,
                )
                for timer in range(num_update_per_opt):
                    _batch, _batch_nxt = self.sample_batch_traj(
                        batch_size=self.batch_size,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )
                    batch = self.unpack_batch(
                        _batch, _batch_nxt, get_perf_action=True,
                        get_latent=True
                    )
                    loss_tp_perf = self.performance.update(
                        batch, timer, update_period=update_period
                    )
                    for i, l in enumerate(loss_tp_perf):
                        if l != 0:
                            loss_perf[i] += [l]

                # Average for losses
                for i, l in enumerate(loss_perf):
                    loss_perf[i] = np.mean(loss_perf[i])

                # Records loss_q, loss_pi, loss_entropy, loss_alpha
                train_records.append(loss_perf)
                latent_mean = self.performance.latent_mean.clone().detach(
                ).cpu().numpy()
                latent_std = self.performance.latent_std.clone().detach().cpu(
                ).numpy()
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
                    wandb.log(
                        {
                            "loss_q (Perf)": loss_perf[0],
                            "loss_pi (Perf)": loss_perf[1],
                            # "loss_entropy (Perf)": loss_perf[2],
                            # "loss_alpha (Perf)": loss_perf[3],
                        },
                        step=self.cnt_step,
                        commit=False
                    )

                # Re-initializes pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, _prev_obs, _prev_obs_stack = self.reset_all_envs(venv)
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

                    progress_perf = self.performance.check(
                        venv,
                        self.cnt_step,
                        states=sample_states,
                        check_type='all_env',
                        verbose=True,
                        num_traj_per_env=num_traj_per_env,
                        revert_task=True,
                        # kwargs for env.simulate_trajectories
                        latent_dist=self.performance.latent_ps,
                        fixed_init=True,
                        traj_size=self.traj_size,
                        frame_skip=self.frame_skip,
                    )

                    # Bound - assume check and update at the same time
                    with torch.no_grad():
                        kl_div = self.performance.get_kl_div().cpu().numpy()
                        renyi_div = self.performance.get_renyi_div().cpu(
                        ).numpy()
                    kl_bound = get_kl_bound(
                        progress_perf[0], kl_div, self.N, self.delta
                    )
                    renyi_bound = get_renyi_bound(
                        progress_perf[0], renyi_div, self.N, self.delta
                    )

                    # Log
                    train_records[-1] = np.insert(
                        train_records[-1], 0, self.cnt_step
                    )
                    train_records[-1] = np.insert(
                        train_records[-1], len(train_records[-1]), kl_bound
                    )
                    train_records[-1] = np.insert(
                        train_records[-1], len(train_records[-1]), renyi_bound
                    )
                    train_progress[0].append(progress_perf)
                    train_progress[1].append([
                        self.cnt_step, kl_bound, renyi_bound
                    ])

                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success": progress_perf[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
                            "buffer traj": len(self.memory),
                            "buffer sample": self.memory.num_sample,
                            "KL bound": kl_bound,
                            "Renyi bound": renyi_bound,
                            "KL div": kl_div,
                            "Renyi div": renyi_div
                        }, step=self.cnt_step, commit=True)
                        if self.use_append_noise:
                            wandb.log({
                                "NU": self.nu,
                            }, step=self.cnt_step, commit=True)

                    # Saves model
                    if save_metric == 'perf':
                        self.save(metric=progress_perf[0])
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
        episode_record = np.array(episode_record)
        return train_records, train_progress, violation_record, episode_record
