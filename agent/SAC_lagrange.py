# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for safety critic learning of fine-tuning phase in SQRL paper.

This file implements safety critic learning for fine-tuning phase in
``Learning to be Safe: Deep RL with a Safety Critic'' (SQRL).

Paper: https://arxiv.org/abs/2010.14603.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler

from agent.SAC_mini import SAC_mini
from agent.model import SACTwinnedQNetwork
from utils.misc import eval_only


class SAC_lagrange(SAC_mini):

    def __init__(
        self, cfg, cfg_arch, cfg_env, cfg_backup, verbose: bool = True
    ):
        """
        Args:
            cfg (object): update-rekated hyper-parameter configuration.
            cfg_arch (object): NN architecture configuration.
            cfg_env (object): environment configuration.
            cfg_backup (object): backup agent configuration.
        """
        super().__init__(cfg, cfg_arch, cfg_env)
        self.cfg_safety = cfg_backup.arch

        # nu-related hyper-parameters
        self.init_nu = cfg.nu
        self.learn_nu = cfg.learn_nu
        self.log_nu = torch.tensor(np.log(self.init_nu)).to(self.device)
        self.lr_nu_schedule = cfg.lr_nu_schedule
        if self.learn_nu:
            self.log_nu.requires_grad = True
            self.lr_nu = cfg.lr_nu
            if self.lr_nu_schedule:
                self.lr_nu_period = cfg.lr_nu_period
                self.lr_nu_decay = cfg.lr_nu_decay
                self.lr_nu_end = cfg.lr_nu_end
        #
        self.safe_thr = cfg_backup.train.threshold
        self.safe_latent_dim = cfg_backup.train.latent_dim
        self.safe_critic_has_act_ind = cfg_backup.arch.critic_has_act_ind

    @property
    def nu(self):
        return self.log_nu.exp()

    def build_network(
        self, verbose: bool = True, actor_path: Optional[str] = None,
        critic_path: Optional[str] = None,
        safety_critic_path: Optional[str] = None
    ):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to True.
            actor_path (str, optional): path to the actor weights. Loads the
                weights is the path is not None. Defaults to None.
            critic_path (str, optional):  path to the critic weights. Loads the
                weights is the path is not None. Defaults to None.
            safety_critic_path (str, optional): path to the safety critic
                weights. Loads the weights is the path is not None. Defaults to
                None.
        """
        super().build_network(
            verbose=verbose, actor_path=actor_path, critic_path=critic_path
        )
        if self.safe_critic_has_act_ind:
            critic_action_dim = self.action_dim + self.act_ind_dim
        else:
            critic_action_dim = self.action_dim

        self.safety_critic = SACTwinnedQNetwork(
            input_n_channel=self.obs_channel, img_sz=[self.img_h, self.img_w],
            latent_dim=self.safe_latent_dim,
            mlp_dim=self.cfg_safety.mlp_dim.critic,
            action_dim=critic_action_dim,
            append_dim=self.cfg_safety.append_dim,
            activation_type=self.cfg_safety.activation.critic,
            kernel_sz=self.cfg_safety.kernel_size,
            stride=self.cfg_safety.stride, n_channel=self.cfg_safety.n_channel,
            use_sm=self.cfg_safety.use_sm, use_ln=self.cfg_safety.use_ln,
            device=self.device, verbose=False
        )
        if safety_critic_path is not None:
            self.safety_critic.load_state_dict(
                torch.load(safety_critic_path, map_location=self.device)
            )
        eval_only(self.safety_critic)

        # Sets up optimizer
        self.build_optimizer()

    def build_optimizer(self):
        """
        Builds an optimizer for the Lagrange multiplier (nu) of the safety
        constraint formulated by the safety critic and the safety threshold.
        """
        super(SAC_lagrange, self).build_optimizer()
        print("Build SAC Lagrangian relaxation optimizer.")
        if self.learn_nu:
            self.log_nu.requires_grad = True
            self.log_nu_optimizer = Adam([self.log_nu], lr=self.lr_nu)
            if self.lr_nu_schedule:
                self.log_nu_scheduler = lr_scheduler.StepLR(
                    self.log_nu_optimizer, step_size=self.lr_nu_period,
                    gamma=self.lr_nu_decay
                )

    # update functions
    def update_nu_hyper_param(self):
        """Updates the learning rate of log nu.
        """
        if self.lr_nu_schedule:
            lr = self.log_nu_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.lr_nu_end:
                for param_group in self.log_nu_optimizer.param_groups:
                    param_group['lr'] = self.lr_nu_end
            else:
                self.log_nu_scheduler.step()

    def update_hyper_param(self):
        """Updates all the hyper-parameters.
        """
        super(SAC_lagrange, self).update_hyper_param()
        if self.learn_nu:
            self.update_nu_hyper_param()

    def update_actor(self, batch) -> Tuple[float, float, float, float]:
        """Updates the actor given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
            float: loss of entropy coefficient (alpha).
            float: loss of Lagrange multiplier of the safety constraint.
        """
        _, _, state, _, _, _, _, append, _, _, _, _, _ = batch

        self.critic.eval()
        self.actor.train()

        action_sample, log_prob = self.actor.sample(
            state, append=append, latent=None, detach_encoder=True
        )
        action_sample_m = action_sample.clone()
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample_m = torch.cat((action_sample, act_ind_rep), dim=-1)
        action_sample_s = action_sample.clone()
        if self.safe_critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample_s = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(
            state, action_sample_m, append=append, latent=None,
            detach_encoder=True
        )
        s_q_pi_1, s_q_pi_2 = self.safety_critic(
            state, action_sample_s, append=append, latent=None,
            detach_encoder=True
        )

        if self.mode == 'RA' or self.mode == 'safety':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)
        max_sq_pi = torch.max(s_q_pi_1, s_q_pi_2)

        # reward case: min_{alpha, nu} max_theta
        #     E[ Q_p - alpha * (log pi + H) + nu * (thr - Q_s)]
        # loss_pi = -Q_p + alpha * log pi + nu (Q_s - thr)
        # loss_alpha = - alpha * (log pi + H)
        # loss_nu = nu (thr - Q_s)
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        loss_safety = max_sq_pi.mean() - self.safe_thr
        if self.mode == 'RA' or self.mode == 'safety':
            loss_pi = loss_q_eval + self.alpha * loss_entropy
        elif self.mode == 'performance':
            loss_pi = (
                -loss_q_eval + self.alpha * loss_entropy
                + self.nu * loss_safety
            )
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Automatic temperature tuning
        loss_alpha = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        if self.learn_alpha:
            self.log_alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.log_alpha_optimizer.step()

        loss_nu = self.nu * (-loss_safety).detach()
        if self.learn_nu:
            self.log_nu_optimizer.zero_grad()
            loss_nu.backward()
            self.log_nu_optimizer.step()
        return (
            loss_pi.item(), loss_entropy.item(), loss_alpha.item(),
            loss_nu.item()
        )

    def update(
        self, batch, timer: int, update_period: int = 2
    ) -> Tuple[float, float, float, float, float]:
        """
        Updates the actor and critic given the sampled transitions from the
        buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
            timer (int): the timer of the number of updates happened in a
                single optimization.
            update_period (int): the period to update the actor and the critic
                target.

        Returns:
            float: loss of critic from the Bellman equation.
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
            float: loss of entropy coefficient (alpha).
            float: loss of Lagrange multiplier of the safety constraint.
        """
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_ent, loss_alpha, loss_nu = 0, 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_ent, loss_alpha, loss_nu = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_ent, loss_alpha, loss_nu
