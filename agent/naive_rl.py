# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for naive policy training.

This file implements the training of the naive policy methods, including Base
(task rewards only), RP (rewards with penalty), and prior of Recovery RL.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import torch
import wandb

from agent.SAC_mini import SAC_mini
from agent.base_training import BaseTraining
from agent.scheduler import StepLR
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class NaiveRL(BaseTraining):

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
        super().__init__(cfg, cfg_env, cfg_train)

        print("= Constructing policy agent")
        self.policy = SAC_mini(cfg_train, cfg_arch, cfg_env)
        self.policy.build_network(verbose=verbose)

        # alias
        self.module_all = [self.policy]
        self.performance = self.policy

        # probability of using backup policy: decaying exponentially to 0 and
        # in the timescale of steps.
        self.use_append_noise = False
        if cfg.use_append_noise:
            self.use_append_noise = True
            self.append_noise_std = torch.tensor([
                cfg.l_x_noise_std, cfg.heading_noise_std
            ], device=self.device)
            self.nu_scheduler = StepLR(
                init_value=cfg.nu, period=cfg.nu_period, decay=cfg.nu_decay,
                end_value=0.
            )
            self.nu = self.nu_scheduler.get_variable()
            self.append_noise_dist = torch.distributions.Normal(
                torch.zeros((2), device=self.device),
                self.nu * self.append_noise_std
            )

    @property
    def has_backup(self):
        return False

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the naive RL policy: single mode and without latent.

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
        s, _ = venv.reset(random_init=random_init)
        z = None

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
                    s, append=append_all, latent=None
                )
                a_all[:, :-1] = a_tmp.clone()
                a_all[:, -1] = self.policy.act_ind

            # Applies action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_all)

            # Gets new append
            append_nxt_all = venv.get_append(venv.get_attr('_state'))

            # Checkss all envs
            for env_ind, (s_, r, done, info) in enumerate(
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
                self.store_transition(
                    z, s[env_ind].unsqueeze(0).to(self.image_device), action,
                    r,
                    s_.unsqueeze(0).to(self.image_device), done, info
                )

                if done:
                    obs, _ = venv.reset_one(
                        env_ind, random_init=random_init, verbose=False
                    )
                    s_all[env_ind] = obs
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

                # Updates critic/actor
                loss = np.zeros(4)
                for timer in range(num_update_per_opt):
                    batch = self.unpack_batch(
                        self.sample_batch(), get_latent=False
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
                s, _ = venv.reset(random_init=random_init)

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

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Sets states for check()
                    sample_states = self.get_check_states(env, num_rnd_traj)
                    if self.cfg.check_type == 'all_env':
                        progress = self.policy.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='all_env',
                            verbose=True,
                            num_traj_per_env=num_traj_per_env,
                            revert_task=True,
                            # kwargs for env.simulate_trajectories
                            latent_dist=None,
                        )
                    else:
                        progress = self.policy.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='random',
                            verbose=True,
                            revert_task=True,
                            sample_task=True,
                            num_rnd_traj=num_rnd_traj,
                        )
                    train_progress[0].append(progress)
                    train_progress[1].append(self.cnt_step)
                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success": progress[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
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

    # overrides restore
    def restore(self, step: int, logs_path: str):
        """Restores the model from the given path.

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
        """
        model_folder = path_c = os.path.join(logs_path)
        path_c = os.path.join(
            model_folder, 'critic', 'critic-{}.pth'.format(step)
        )
        path_a = os.path.join(
            model_folder, 'actor', 'actor-{}.pth'.format(step)
        )

        self.policy.critic.load_state_dict(
            torch.load(path_c, map_location=self.device)
        )
        self.policy.critic.to(self.device)
        self.policy.critic_target.load_state_dict(
            torch.load(path_c, map_location=self.device)
        )
        self.policy.critic_target.to(self.device)
        self.policy.actor.load_state_dict(
            torch.load(path_a, map_location=self.device)
        )
        self.policy.actor.to(self.device)
        print(
            '  <= Restore policy with {} updates from {}.'.format(
                step, model_folder
            )
        )
