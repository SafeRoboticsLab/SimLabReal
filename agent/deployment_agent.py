# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for loading actor and critic for deployment.

This file loads all variants of actors and critics trained in this project for
the deployment in the real world.
"""

from typing import Union
import os
import numpy as np
import torch
import glob

from agent.base_SAC import BaseSAC
from agent.model import SACPiNetwork, SACTwinnedQNetwork


class DeploymentAgent(BaseSAC):

    def __init__(self, cfg, cfg_arch, cfg_env) -> None:
        self.cfg = cfg
        self.cfg_arch = cfg_arch
        self.device = cfg.device
        self.mode = cfg.mode
        self.eval = True

        # == ENV PARAM ==
        self.obs_channel = cfg_env.obs_channel
        self.action_mag = cfg_env.action_mag
        self.action_dim = cfg_env.action_dim
        self.img_w = cfg_env.img_w
        self.img_h = cfg_env.img_h

        self.critic_has_act_ind = cfg_arch.critic_has_act_ind
        if cfg_arch.act_ind is not None:
            self.act_ind = torch.FloatTensor(cfg_arch.act_ind).to(self.device)
            self.act_ind_dim = self.act_ind.shape[0]

        if self.cfg.has_latent:
            self.latent_dim = self.cfg.latent_dim
            self.latent_dim_cnn = self.cfg.latent_dim_cnn
        else:
            self.latent_dim = 0
            self.latent_dim_cnn = 0

    def build_network(self, verbose: bool = False):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to False.
        """

        if self.critic_has_act_ind:
            critic_action_dim = self.action_dim + self.act_ind_dim
        else:
            critic_action_dim = self.action_dim

        self.critic = SACTwinnedQNetwork(
            input_n_channel=self.obs_channel,
            img_sz=[self.img_h, self.img_w],
            latent_dim=self.latent_dim,
            latent_dim_cnn=self.latent_dim_cnn,
            action_dim=critic_action_dim,
            mlp_dim=self.cfg_arch.mlp_dim.critic,
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
        self.critic_target = self.critic  #! dummy usage

        # Loads model if specified.
        if self.cfg.critic_path is not None:
            print("Load critic weights from: ", self.cfg.critic_path)
            self.critic.load_state_dict(
                torch.load(self.cfg.critic_path, map_location=self.device)
            )

        if self.cfg.has_actor:
            self.actor = SACPiNetwork(
                input_n_channel=self.obs_channel,
                img_sz=[self.img_h, self.img_w],
                latent_dim=self.latent_dim,
                latent_dim_cnn=self.latent_dim_cnn,
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

            # Loads model if specified.
            if self.cfg.actor_path is not None:
                print("Load actor weights from: ", self.cfg.actor_path)
                self.actor.load_state_dict(
                    torch.load(self.cfg.actor_path, map_location=self.device)
                )

        self._latent_dist = None
        self._has_latent = False
        if self.cfg.has_latent:
            self._has_latent = True
            if self.cfg.uses_latent_prior:
                _std = self.cfg_arch.latent_prior_std
                # prior
                latent_mean = torch.zeros((self.latent_dim))
                latent_std = _std * torch.ones(self.latent_dim)
            else:
                print("Load latent distribution from: ", self.cfg.latent_path)
                latent_dict = torch.load(
                    self.cfg.latent_path, map_location=self.device
                )
                latent_mean = latent_dict["latent_mean"]
                latent_std = latent_dict["latent_std"]

            self.latent_mean = latent_mean.to(self.device)
            self.latent_std = latent_std.to(self.device)
            self._latent_dist = torch.distributions.Normal(
                self.latent_mean, self.latent_std
            )

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
        if self.has_latent:
            return self._value_with_latent(obs, append)
        else:
            return self._value_no_latent(obs, append)

    def _value_no_latent(
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
        u = self.actor(obs, append=append)
        u = torch.from_numpy(u).to(self.device)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(u.shape[0], 1)
            u = torch.cat((u, act_ind_rep), dim=-1)
        v = self.critic(obs, u, append=append)[0]
        if len(obs.shape) == 3:
            v = v[0]
        return v

    def _value_with_latent(
        self, obs: Union[np.ndarray, torch.Tensor], append: Union[np.ndarray,
                                                                  torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Returns the critic evaluation at this observation and append info.

        In addition to the observation and append information, it samples
        latent variables from the distribution to obtain latent-conditioned
        critic values.

        Args:
            obs (np.ndarray or torch.Tensor): interested observation.
            append (np.ndarray or torch.Tensor): extra info to be appended to
                the input to the critic.

        Returns:
            np.ndarray or torch.Tensor: critic values.
        """
        if len(obs.shape) == 3:
            num_obs = 1
        else:
            num_obs = obs.shape[0]
        z = self.latent_mean.clone().view(1, -1).tile((num_obs, 1))
        u = self.actor(obs, append=append, latent=z)
        u = torch.from_numpy(u).to(self.device)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(u.shape[0], 1)
            u = torch.cat((u, act_ind_rep), dim=-1)
        v = self.critic(obs, u, append=append, latent=z)[0]
        if len(obs.shape) == 3:
            v = v[0]
        return v

    @property
    def has_latent(self):
        """Whether the policy is latent-conditioned."""
        return self._has_latent

    @property
    def latent_dist(self):
        """The distribution of the latent variables."""
        return self._latent_dist

    #! Not used. Only for inheritance.
    def update_actor(self, batch):
        pass

    def update_critic(self, batch):
        pass

    def update(self, batch, timer: int, update_period: int = 2):
        pass