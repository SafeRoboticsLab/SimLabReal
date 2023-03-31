# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for soft actor-critic variants.

This file implements a parent class for all soft actor-critic (SAC) variants
used in this project. A child class should implement the following functions
    + has_latent: returns True if actor and critic is latent-conditioned.
    + latent_dist: the distribution of the latent variables.
    + update / update_actor / update_critic: customized functions to update the
        actor or the critic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import os
import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler

from env.vec_env import VecEnvWrapper
from agent.model import SACPiNetwork, SACTwinnedQNetwork
from agent.scheduler import StepLRMargin
from utils.misc import soft_update, save_model, optimizer_to


class BaseSAC(ABC):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            cfg (object): update-related hyper-parameter cfguration.
            cfg_arch (object): NN architecture cfguration.
            cfg_env (object): environment cfguration.
        """
        self.cfg = cfg
        self.cfg_arch = cfg_arch
        self.eval = cfg.eval

        # == env param ==
        self.obs_channel = cfg_env.obs_channel
        self.action_mag = cfg_env.action_mag
        self.action_dim = cfg_env.action_dim
        self.img_w = cfg_env.img_w
        self.img_h = cfg_env.img_h

        # NN: device, action indicators
        self.device = cfg.device
        self.critic_has_act_ind = cfg_arch.critic_has_act_ind
        if cfg_arch.act_ind is not None:
            self.act_ind = torch.FloatTensor(cfg_arch.act_ind).to(self.device)
            self.act_ind_dim = self.act_ind.shape[0]

        # reach-avoid setting
        self.mode = cfg.mode

        # == param for training ==
        if not self.eval:
            self.terminal_type = cfg.terminal_type

            # nn
            self.batch_size = cfg.batch_size

            # learning rate
            self.lr_a_schedule = cfg.lr_a_schedule
            self.lr_c_schedule = cfg.lr_c_schedule
            if self.lr_a_schedule:
                self.lr_a_period = cfg.lr_a_period
                self.lr_a_decay = cfg.lr_a_decay
                self.lr_a_end = cfg.lr_a_end
            if self.lr_c_schedule:
                self.lr_c_period = cfg.lr_c_period
                self.lr_c_decay = cfg.lr_c_decay
                self.lr_c_end = cfg.lr_c_end
            self.lr_c = cfg.lr_c
            self.lr_a = cfg.lr_a

            # Discount factor
            self.gamma_schedule = cfg.gamma_schedule
            if self.gamma_schedule:
                self.gamma_scheduler = StepLRMargin(
                    init_value=cfg.gamma, period=cfg.gamma_period,
                    decay=cfg.gamma_decay, end_value=cfg.gamma_end,
                    goal_value=1.
                )
                self.gamma = self.gamma_scheduler.get_variable()
            else:
                self.gamma = cfg.gamma

            # Target Network Update
            self.tau = cfg.tau

            # alpha-related hyper-parameters
            self.init_alpha = cfg.alpha
            self.learn_alpha = cfg.learn_alpha
            self.log_alpha = torch.tensor(np.log(self.init_alpha)
                                         ).to(self.device)
            self.target_entropy = -self.action_dim
            if self.learn_alpha:
                self.log_alpha.requires_grad = True
                self.lr_al = cfg.lr_al
                print(
                    "SAC with learnable alpha and target entropy = {:.1e}".
                    format(self.target_entropy)
                )
            else:
                print("SAC with fixed alpha = {:.1e}".format(self.init_alpha))

    @property
    def alpha(self):
        """Entropy coefficient in SAC."""
        return self.log_alpha.exp()

    @property
    @abstractmethod
    def has_latent(self) -> bool:
        """Whether the policy is latent-conditioned."""
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dist(self):
        """The distribution of the latent variables."""
        raise NotImplementedError

    def build_network(
        self, verbose=True, actor_path=None, critic_path=None, tie_conv=True
    ):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to True.
            actor_path (str, optional): path to the actor weights. Loads the
                weights is the path is not None. Defaults to None.
            critic_path (str, optional):  path to the critic weights. Loads the
                weights is the path is not None. Defaults to None.
            tie_conv (bool, optional): ties the actor's encoder with the
                critic's encoder if True. Defaults to True.
        """
        if self.critic_has_act_ind:
            critic_action_dim = self.action_dim + self.act_ind_dim
        else:
            critic_action_dim = self.action_dim

        self.critic = SACTwinnedQNetwork(
            input_n_channel=self.obs_channel,
            img_sz=[self.img_h, self.img_w],
            latent_dim=self.cfg.latent_dim,
            latent_dim_cnn=self.cfg.latent_dim_cnn,
            mlp_dim=self.cfg_arch.mlp_dim.critic,
            action_dim=critic_action_dim,
            append_dim=self.cfg_arch.append_dim,
            activation_type=self.cfg_arch.activation.critic,
            kernel_sz=self.cfg_arch.kernel_size,
            stride=self.cfg_arch.stride,
            n_channel=self.cfg_arch.n_channel,
            use_sm=self.cfg_arch.use_sm,
            use_ln=self.cfg_arch.use_ln,
            device=self.device,
            verbose=verbose,
        )
        if verbose:
            print("\nThe actor shares the same encoder with the critic.")
        self.actor = SACPiNetwork(
            input_n_channel=self.obs_channel,
            img_sz=[self.img_h, self.img_w],
            latent_dim=self.cfg.latent_dim,
            latent_dim_cnn=self.cfg.latent_dim_cnn,
            action_dim=self.action_dim,
            action_mag=self.action_mag,
            mlp_dim=self.cfg_arch.mlp_dim.actor,
            append_dim=self.cfg_arch.append_dim,
            activation_type=self.cfg_arch.activation.actor,
            kernel_sz=self.cfg_arch.kernel_size,
            stride=self.cfg_arch.stride,
            n_channel=self.cfg_arch.n_channel,
            use_sm=self.cfg_arch.use_sm,
            use_ln=self.cfg_arch.use_ln,
            device=self.device,
            verbose=verbose,
        )

        # Load model if specified
        if critic_path is not None:
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
            print("--> Load critic wights from {}".format(critic_path))

        if actor_path is not None:
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            print("--> Load actor wights from {}".format(actor_path))

        # Copy for critic targer
        self.critic_target = copy.deepcopy(self.critic)

        # Tie weights for conv layers
        if tie_conv:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

    def build_optimizer(self):
        """Builds optimizers for NNs and schedulers for learning rates.
        """
        print("Build basic optimizers.")
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_c)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_a)
        if self.lr_c_schedule:
            self.critic_scheduler = lr_scheduler.steplr(
                self.critic_optimizer, step_size=self.lr_c_period,
                gamma=self.lr_c_decay
            )
        if self.lr_a_schedule:
            self.actor_scheduler = lr_scheduler.steplr(
                self.actor_optimizer, step_size=self.lr_a_period,
                gamma=self.lr_a_decay
            )
        if self.learn_alpha:
            self.log_alpha_optimizer = Adam([self.log_alpha], lr=self.lr_al)

    # update functions
    def update_critic_hyper_param(self):
        """Updates the learning rate and the discount rate of critic.
        """
        if self.lr_c_schedule:
            lr = self.critic_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.lr_c_end:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.lr_c_end
            else:
                self.critic_scheduler.step()
        if self.gamma_schedule:
            self.gamma_scheduler.step()
            self.gamma = self.gamma_scheduler.get_variable()

    def update_actor_hyper_param(self):
        """Updates the learning rate of actor.
        """
        if self.lr_a_schedule:
            lr = self.actor_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.lr_a_end:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.lr_a_end
            else:
                self.actor_scheduler.step()

    def update_hyper_param(self):
        """Updates all the hyper-parameters.
        """
        self.update_critic_hyper_param()
        self.update_actor_hyper_param()

    def update_target_networks(self):
        """Updates the target critic network.
        """
        soft_update(self.critic_target, self.critic, self.tau)

    @abstractmethod
    def update_actor(self, batch):
        """Updates the actor given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def update_critic(self, batch):
        """Updates the critic given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, batch, timer: int, update_period: int = 2):
        """
        Updates the actor and critic given the sampled transitions from the
        buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
            timer (int): the timer of the number of updates happened in a
                single optimization.
            update_period (int): the period to update the actor and the critic
                target.
        """
        raise NotImplementedError

    # utils
    @abstractmethod
    def value(
        self, obs: Union[np.ndarray, torch.Tensor], append: Union[np.ndarray,
                                                                  torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns the critic evaluation at this observation and append info.

        Args:
            obs (np.ndarray or torch.Tensor): interested observation.
            append (np.ndarray or torch.Tensor): extra info to be appended to
                the input to the critic.

        Returns:
            np.ndarray or torch.Tensor: critic values.
        """
        raise NotImplementedError

    def save(self, step: int, logs_path: str, max_model: Optional[int] = None):
        """Saves the current actor and critic.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
            max_model (int, optional): the maximum number of models to save.
                Defaults to None.
        """
        path_c = os.path.join(logs_path, 'critic')
        path_a = os.path.join(logs_path, 'actor')
        save_model(self.critic, step, path_c, 'critic', max_model)
        save_model(self.actor, step, path_a, 'actor', max_model)

    def remove(self, step: int, logs_path: str):
        """Removes the actor and critic checkpoints.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
        """
        path_c = os.path.join(
            logs_path, 'critic', 'critic-{}.pth'.format(step)
        )
        path_a = os.path.join(logs_path, 'actor', 'actor-{}.pth'.format(step))
        print("Remove", path_a)
        print("Remove", path_c)
        if os.path.exists(path_c):
            os.remove(path_c)
        if os.path.exists(path_a):
            os.remove(path_a)

    def move_all_to_cpu(self):
        """Moves all the models and tensors to cpu.
        """
        if hasattr(self, 'actor_optimizer'):
            self.actor_optimizer.zero_grad()
            optimizer_to(self.actor_optimizer, 'cpu')
        if hasattr(self, 'critic_optimizer'):
            self.critic_optimizer.zero_grad()
            optimizer_to(self.critic_optimizer, 'cpu')
        self.actor.to('cpu')
        self.critic.to('cpu')
        self.critic_target.to('cpu')
        self.act_ind = self.act_ind.to('cpu')
        if hasattr(self, 'log_alpha'):
            self.log_alpha = self.log_alpha.to('cpu')
            if self.learn_alpha:
                self.log_alpha = self.log_alpha.detach()
        self._prev_device = self.device
        self.device = 'cpu'

    def move_all_to_prev_device(self):
        """Moves all the models and tensors to the previous device.
        """
        self.actor.to(self._prev_device)
        self.critic.to(self._prev_device)
        self.critic_target.to(self._prev_device)
        if hasattr(self, 'actor_optimizer'):
            optimizer_to(self.actor_optimizer, self._prev_device)
        if hasattr(self, 'critic_optimizer'):
            optimizer_to(self.critic_optimizer, self._prev_device)
        self.act_ind = self.act_ind.to(self._prev_device)
        if hasattr(self, 'log_alpha'):
            self.log_alpha = self.log_alpha.to(self._prev_device)
            if self.learn_alpha:
                self.log_alpha.requires_grad = True
                self.log_alpha_optimizer = Adam(
                    [self.log_alpha], lr=self.lr_al
                )  # not ideal since this resets optimizer
        self.device = self._prev_device

    def check(
        self, env: VecEnvWrapper, cnt_step, states, check_type, verbose,
        backup=None, **kwargs
    ):
        """Checks the performance of the actor and critic.

        This serializes the policies and send them to each worker.

        Args:
            env (object): the environment where it evaluates the actor and
                critic .
            cnt_step (int): the current number of training steps.
            states (np.ndarray): initial states where the evaluation rollouts
                start.
            check_type (str): the type of checking.
            verbose (bool, optional): prints messages if True. Defaults to
                True.

        Returns:
            np.ndarray: evaluation stats.
        """

        if self.mode == 'safety' or self.mode == 'risk':
            end_type = 'fail'
        elif self.mode == 'RA':
            end_type = 'safety_ra'
        else:
            end_type = 'TF'

        self.actor.eval()
        self.critic.eval()

        # switch device to cpu
        self.move_all_to_cpu()
        if backup is not None:
            backup.actor.eval()
            backup.critic.eval()
            backup.move_all_to_cpu()

        if check_type == 'random':
            results = env.simulate_trajectories(
                self, mode=self.mode, states=states, end_type=end_type,
                backup=backup, **kwargs
            )[1]
        elif check_type == 'all_env':
            results = env.simulate_all_envs(
                self, mode=self.mode, states=states, end_type=end_type,
                backup=backup, **kwargs
            )[1]
        else:
            raise ValueError(
                "Check type ({}) not supported!".format(check_type)
            )
        if self.mode == 'safety' or self.mode == 'risk':
            failure = np.sum(results == -1) / results.shape[0]
            success = 1 - failure
            train_progress = np.array([success, failure])
        else:
            success = np.sum(results == 1) / results.shape[0]
            failure = np.sum(results == -1) / results.shape[0]
            unfinish = np.sum(results == 0) / results.shape[0]
            train_progress = np.array([success, failure, unfinish])

        if verbose:
            if not self.eval:
                print(
                    '\n{} policy after [{}] steps:'.format(
                        self.mode, cnt_step
                    )
                )
                print(
                    '  - gamma={:.6f}, alpha={:.1e}.'.format(
                        self.gamma, self.alpha
                    )
                )
            if self.mode == 'safety' or self.mode == 'risk':
                print('  - success/failure ratio:', end=' ')
            else:
                print('  - success/failure/unfinished ratio:', end=' ')
            with np.printoptions(formatter={'float': '{: .2f}'.format}):
                print(train_progress)

        # Switch device back
        self.move_all_to_prev_device()
        if backup is not None:
            backup.move_all_to_prev_device()
            backup.actor.train()
            backup.critic.train()

        self.actor.train()
        self.critic.train()
        return train_progress
