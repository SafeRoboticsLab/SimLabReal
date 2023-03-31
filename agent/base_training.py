# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for training agents.

This file implements a parent class for all training agents.
"""

from typing import Optional
from abc import ABC, abstractmethod
from collections import namedtuple
from queue import PriorityQueue
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from agent.replay_memory import ReplayMemory
from agent.scheduler import StepLR
from env.base_env import BaseEnv
from env.vec_env import VecEnvWrapper

TransitionLatent = namedtuple(
    'TransitionLatent', ['z', 's', 'a', 'r', 's_', 'done', 'info']
)


class BaseTraining(ABC):

    def __init__(self, cfg, cfg_env, cfg_train):
        """
        Args:
            cfg (dict): config
            cfg_env (dict): config for environment
            cfg_train (dict): config for training
        """
        super(BaseTraining, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.image_device = cfg.image_device
        self.n_envs = cfg.num_cpus
        self.action_dim = cfg_env.action_dim
        self.max_step_train = cfg_env.max_step_train
        self.max_step_eval = cfg_env.max_step_eval
        self.obs_buffer = cfg_env.obs_buffer
        self.num_visualize_task = cfg.num_visualize_task

        # We assume backup and performance use the same parameters.
        self.batch_size = cfg_train.batch_size
        self.update_period = cfg_train.update_period

        # memory
        self.memory = ReplayMemory(cfg.memory_capacity, cfg.seed)
        self.rng = np.random.default_rng(seed=cfg.seed)

        # saving models
        self.save_top_k = self.cfg.save_top_k
        self.pq_top_k = PriorityQueue()

        # probability of adding noise to append: decaying exponentially to 0
        # and in the timescale of steps.
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
            print(self.device)
            self.append_noise_dist = torch.distributions.Normal(
                torch.zeros((2), device=self.device),
                self.nu * self.append_noise_std
            )

    @property
    @abstractmethod
    def has_backup(self):
        """Whether there is a back policy."""
        raise NotImplementedError

    def sample_batch(
        self, batch_size: Optional[int] = None,
        recent_size: Optional[int] = None
    ):
        """Samples a batch of transitions.

        Args:
            batch_size (int, optional): batch size. Defaults to None.
            recent_size (int, optional): sample from the recent_size most
                recent transitions. Defaults to None.
        """
        if batch_size is None:
            batch_size = self.batch_size
        transitions, _ = self.memory.sample(batch_size, recent_size)
        batch = TransitionLatent(*zip(*transitions))
        return batch

    def store_transition(self, *args):
        """Stores a transition."""
        self.memory.update(TransitionLatent(*args))

    def unpack_batch(
        self, batch: TransitionLatent, get_latent: bool = True,
        get_perf_action: bool = False, get_l_x_ra: bool = False,
        get_latent_backup: bool = False
    ):
        """Unpacks a batch of transitions.

        Args:
            batch (TransitionLatent): a batch of transitions.
            get_latent (bool): whether to return latent. Defaults to True.
            get_perf_action (bool): whether to return performance action.
                Defaults to False.
            get_l_x_ra (bool): whether to return l_x and ra. Defaults to False.
            get_latent_backup (bool): whether to return latent of backup
                policy. Defaults to False.
        """
        non_final_mask = torch.tensor(
            tuple(map(lambda s: not s, batch.done)), dtype=torch.bool
        ).view(-1).to(self.device)
        non_final_state_nxt = torch.cat([
            s for done, s in zip(batch.done, batch.s_) if not done
        ]).to(self.device)
        state = torch.cat(batch.s).to(self.device)
        reward = torch.FloatTensor(batch.r).to(self.device)
        g_x = torch.FloatTensor([info['g_x'] for info in batch.info]
                               ).to(self.device).view(-1)

        if get_perf_action:  # recovery RL separates a_shield and a_perf.
            if batch.info[0]['a_perf'].dim() == 1:
                action = torch.FloatTensor([
                    info['a_perf'] for info in batch.info
                ])
            else:
                action = torch.cat([info['a_perf'] for info in batch.info])
            action = action.to(self.device)
        else:
            action = torch.cat(batch.a).to(self.device)

        latent = None
        if get_latent:
            if get_latent_backup:
                latent = torch.cat([info['z_backup'] for info in batch.info])
                latent = latent.to(self.device)
            else:
                latent = torch.cat(batch.z).to(self.device)

        append = torch.cat([info['append'] for info in batch.info]
                          ).to(self.device)
        non_final_append_nxt = torch.cat([
            info['append_nxt'] for info in batch.info
        ]).to(self.device)[non_final_mask]

        l_x_ra = None
        if get_l_x_ra:
            l_x_ra = torch.FloatTensor([info['l_x_ra'] for info in batch.info])
            l_x_ra = l_x_ra.to(self.device).view(-1)

        binary_cost = torch.FloatTensor([
            info['binary_cost'] for info in batch.info
        ])
        binary_cost = binary_cost.to(self.device).view(-1)

        hn = None
        cn = None
        return (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x,
            latent, append, non_final_append_nxt, l_x_ra, binary_cost, hn, cn
        )

    def unpack_single_task_from_batch(self, batch: TransitionLatent):
        """Gets a single task from the transition batch.

        Args:
            batch (TransitionLatent): a batch of transitions.
        """
        latent = batch.z[0]
        task_id = batch.info[0]['task_id']
        state = batch.info[0]['state']
        return latent, task_id, state

    def save(self, metric: Optional[float] = None, force_save: bool = False):
        """Saves the model.

        Args:
            metric (float): the metric to save the model
            force_save (bool): whether to force save the model
        """
        assert metric is not None or force_save, \
            "should provide metric of force save"
        save_current = False
        if force_save:
            save_current = True
        elif self.pq_top_k.qsize() < self.save_top_k:
            self.pq_top_k.put((metric, self.cnt_step))
            save_current = True
        elif metric > self.pq_top_k.queue[0][0]:  # overwrite
            # Remove old one
            _, step_remove = self.pq_top_k.get()
            for module, module_folder in zip(
                self.module_all, self.module_folder_all
            ):
                module.remove(int(step_remove), module_folder)
            self.pq_top_k.put((metric, self.cnt_step))
            save_current = True

        if save_current:
            print('\nSaving current model...')
            for module, module_folder in zip(
                self.module_all, self.module_folder_all
            ):
                module.save(self.cnt_step, module_folder)
            print(self.pq_top_k.queue)

    def restore(
        self, step: int, logs_path: str, agent_type: str,
        actor_path: Optional[str] = None
    ):
        """Restores the weights of the neural network.

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
            agent_type (str): performance or backup.
            actor_path (str): the path of the actor model. If None, use the
                default path.
        """
        model_folder = path_c = os.path.join(logs_path, agent_type)
        path_c = os.path.join(
            model_folder, 'critic', 'critic-{}.pth'.format(step)
        )
        if actor_path is not None:
            path_a = actor_path
        else:
            path_a = os.path.join(
                model_folder, 'actor', 'actor-{}.pth'.format(step)
            )
        if agent_type == 'backup':
            self.backup.critic.load_state_dict(
                torch.load(path_c, map_location=self.device)
            )
            self.backup.critic.to(self.device)
            self.backup.critic_target.load_state_dict(
                torch.load(path_c, map_location=self.device)
            )
            self.backup.critic_target.to(self.device)
            self.backup.actor.load_state_dict(
                torch.load(path_a, map_location=self.device)
            )
            self.backup.actor.to(self.device)
        elif agent_type == 'performance':
            self.performance.critic.load_state_dict(
                torch.load(path_c, map_location=self.device)
            )
            self.performance.critic.to(self.device)
            self.performance.critic_target.load_state_dict(
                torch.load(path_c, map_location=self.device)
            )
            self.performance.critic_target.to(self.device)
            self.performance.actor.load_state_dict(
                torch.load(path_a, map_location=self.device)
            )
            self.performance.actor.to(self.device)
        print(
            '  <= Restore {} with {} updates from {}.'.format(
                agent_type, step, model_folder
            )
        )

    def get_check_states(self, env: BaseEnv, num_rnd_traj: int):
        """Gets the states to check the performance of the policy.

        Args:
            env (BaseEnv): the environment class.
            num_rnd_traj (int): the number of trajectories to sample.
        """
        if env.env_type == 'advanced':
            sample_states = None
        else:
            x_range = [0.1, 0.1]
            y_range = [-0.5, 0.5]
            theta_range = [-np.pi / 3, np.pi / 3]
            sample_x = self.rng.uniform(
                x_range[0], x_range[1], (num_rnd_traj, 1)
            )
            sample_y = self.rng.uniform(
                y_range[0], y_range[1], (num_rnd_traj, 1)
            )
            sample_theta = self.rng.uniform(
                theta_range[0], theta_range[1], (num_rnd_traj, 1)
            )
            sample_states = np.concatenate((sample_x, sample_y, sample_theta),
                                           axis=1)
        return sample_states

    def get_figures(
        self, venv: VecEnvWrapper, env: BaseEnv, plot_v: bool, vmin: float,
        vmax: float, save_figure: bool, plot_figure: bool,
        figure_folder_perf: str, plot_backup: bool = False,
        figure_folder_backup: Optional[str] = None, plot_shield: bool = False,
        plot_shield_value: bool = False,
        figure_folder_shield: Optional[str] = None,
        shield_dict: Optional[dict] = None, latent_dist=None, **kwargs
    ):
        """Gets the figures of the policy.

        Args:
            venv (VecEnv): the vectorized environment class.
            env (BaseEnv): the environment class.
            plot_v (bool): whether to plot the value function.
            vmin (float): the minimum value of the value function.
            vmax (float): the maximum value of the value function.
            save_figure (bool): whether to save the figures.
            plot_figure (bool): whether to plot the figures.
            figure_folder_perf (str): the folder to save the figures of the
                performance policy.
            plot_backup (bool): whether to plot the value function of the
                backup policy.
            figure_folder_backup (str): the folder to save the figures of the
                backup policy.
            plot_shield (bool): whether to plot the shield function.
            plot_shield_value (bool): whether to plot the shield value.
            figure_folder_shield (str): the folder to save the figures of the
                shield.
            shield_dict (dict): the dictionary of the shield.
            latent_dist (LatentDist): the latent distribution.
            **kwargs: other arguments.
        """
        if latent_dist is not None:
            perf_latent_dist = latent_dist
            backup_latent_dist = latent_dist
        else:
            perf_latent_dist = self.performance.latent_dist
            if plot_backup:
                backup_latent_dist = self.backup.latent_dist

        for task_ind in range(self.num_visualize_task):
            print('Visualizing task {}...'.format(task_ind))
            if task_ind == 0:
                plot_v_task = True
            else:
                plot_v_task = plot_v
            task, task_id = venv.sample_task(return_id=True)
            perf_mode = self.performance.mode
            if perf_mode == 'safety' or perf_mode == 'risk':
                end_type = 'fail'
                plot_contour = True
            elif perf_mode == 'RA':
                end_type = 'safety_ra'
                plot_contour = True
            else:
                end_type = 'TF'
                plot_contour = False

            fig_perf = env.visualize(
                self.performance.value, self.performance,
                mode=self.performance.mode, end_type=end_type,
                latent_dist=perf_latent_dist, plot_v=plot_v_task, vmin=vmin,
                vmax=vmax, cmap='seismic', normalize_v=True,
                plot_contour=plot_contour, task=task, revert_task=False,
                **kwargs
            )
            if plot_backup:
                fig_backup = env.visualize(
                    self.backup.value, self.backup, mode=self.backup.mode,
                    end_type='fail', latent_dist=backup_latent_dist,
                    plot_v=plot_v_task, vmin=vmin, vmax=vmax, cmap='seismic',
                    normalize_v=False, plot_contour=True, task=task,
                    revert_task=False, **kwargs
                )
            if plot_shield:
                assert self.has_backup, (
                    "This figure requires policy with shielding scheme."
                )
                fig_shield = env.visualize(
                    self.value, self.performance, mode=self.performance.mode,
                    end_type='TF', latent_dist=perf_latent_dist,
                    plot_v=plot_shield_value, vmin=vmin, vmax=vmax,
                    cmap='seismic', normalize_v=False, plot_contour=False,
                    task=task, revert_task=False, shield=True,
                    backup=self.backup, shield_dict=shield_dict, **kwargs
                )
            if save_figure:
                fig_perf.savefig(
                    os.path.join(
                        figure_folder_perf, '{:d}_{:d}_{:d}.png'.format(
                            self.cnt_step, task_id, int(plot_v_task)
                        )
                    )
                )
                if plot_backup:
                    fig_backup.savefig(
                        os.path.join(
                            figure_folder_backup, '{:d}_{:d}_{:d}.png'.format(
                                self.cnt_step, task_id, int(plot_v_task)
                            )
                        )
                    )
                if plot_shield:
                    fig_shield.savefig(
                        os.path.join(
                            figure_folder_shield, '{:d}_{:d}_{:d}.png'.format(
                                self.cnt_step, task_id, int(plot_v_task)
                            )
                        )
                    )
                plt.close('all')
            if plot_figure:
                fig_perf.show()
                if plot_backup:
                    fig_backup.show()
                if plot_shield:
                    fig_shield.show()
                plt.pause(0.01)
                plt.close()

    @abstractmethod
    def learn(self):
        """Learns the policy.
        """
        raise NotImplementedError

    def value(self, obs: np.ndarray, append: np.ndarray):
        """
        Gets safety values from backup critic using actions from the
        performance actor.

        Args:
            obs (np.ndarray): observation.
            append (np.ndarray): extra information.

        Returns:
            np.ndarray: values.
        """
        assert self.has_backup, (
            "This value function only supports policy "
            + "with shielding scheme."
        )
        if len(obs.shape) == 3:
            num_obs = 1
        else:
            num_obs = obs.shape[0]
        z_perf = None
        if self.performance.has_latent:
            z_perf = (
                self.performance.latent_mean.clone().view(1, -1).tile(
                    (num_obs, 1)
                )
            )
        u = self.performance.actor(obs, append, latent=z_perf)
        u = torch.from_numpy(u).to(self.device)
        if self.backup.critic_has_act_ind:
            act_ind_rep = self.performance.act_ind.repeat(u.shape[0], 1)
            u = torch.cat((u, act_ind_rep), dim=-1)

        z_backup = None
        if self.backup.has_latent:
            z_backup = (
                self.backup.latent_mean.clone().view(1, -1).tile((num_obs, 1))
            )

        # Truncate obs if mixed stack - a bit hacky
        if self.backup.obs_channel == 3:
            obs = obs[:, -3:]
        v = self.backup.critic(obs, u, append, latent=z_backup)[0]
        if len(obs.shape) == 3:
            v = v[0]
        return v
