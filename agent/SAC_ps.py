# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for soft actor-critic variants with posterior distribution.

This file implements a parent class for all soft actor-critic (SAC) variants
with a trainable latent distribution in this project (especially posterior).
Specifically, this class serves as a parent class for SAC_ps_c and SAC_ps_noc.
"""

from typing import Tuple, Optional
import os
import torch
from torch.optim import Adam

from agent.SAC_latent import SAC_latent
from utils.misc import eval_only
from utils.dist import (
    std2logvar, logvar2std, logvar2baselogvar, baselogvar2logvar
)


class SAC_ps(SAC_latent):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            cfg (Class object): training hyper-parameters.
            cfg_arch (Class object): NN architecture hyper-parameters.
            cfg_env (Class object): environment hyper-parameters.
        """
        super().__init__(cfg, cfg_arch, cfg_env)

        # Latent - Assume prior is Gaussian with diagonal covariances
        self.latent_prior_mean = torch.zeros((self.latent_dim)).to(self.device)
        self.latent_prior_std = cfg.latent_prior_std * torch.ones(
            (self.latent_dim)
        ).to(self.device)
        self.latent_prior = torch.distributions.Normal(
            self.latent_prior_mean, self.latent_prior_std
        )

        # Initialize posterior
        self.latent_mean = torch.zeros((self.latent_dim), device=self.device)
        self.latent_base_logvar = torch.zeros((self.latent_dim),
                                              device=self.device)
        logvar = baselogvar2logvar(
            self.latent_base_logvar, self.latent_prior_std[0]
        )
        self.latent_std = logvar2std(logvar)
        self.latent_ps = torch.distributions.Normal(
            self.latent_mean, self.latent_std
        )

        # Latent optimizer
        self.lr_lm = cfg.lr_lm
        self.lr_ll = cfg.lr_ll
        self.div_weight = cfg.div_weight
        self.bound_type = cfg.bound_type
        self.latent_optimizer_state = None

    @property
    def latent_dist(self):
        return self.latent_ps

    def build_latent_optimzer(self):
        """Builds optimizers for the mean and the variance of the posterior.
        """
        # Make a copy of posterior for optimization
        self.latent_mean_copy = self.latent_mean.clone().detach()
        self.latent_base_logvar_copy = self.latent_base_logvar.clone().detach()
        self.latent_mean_copy.requires_grad = True
        self.latent_base_logvar_copy.requires_grad = True

        logvar = baselogvar2logvar(
            self.latent_base_logvar_copy, self.latent_prior_std[0]
        )
        self.latent_std_copy = logvar2std(logvar)
        self.latent_ps_copy = torch.distributions.Normal(
            self.latent_mean_copy, self.latent_std_copy
        )

        self.latent_optimizer = Adam([
            {
                'params': self.latent_mean_copy,
                'lr': self.lr_lm
            },
            {
                'params': self.latent_base_logvar_copy,
                'lr': self.lr_ll
            },
        ])
        if self.latent_optimizer_state is not None:
            self.latent_optimizer.load_state_dict(self.latent_optimizer_state)

    def build_network(
        self, verbose: bool = True, actor_path: Optional[str] = None,
        critic_path: Optional[str] = None
    ):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool, optional): prints out meassages if Ture. Defaults to
                True.
            actor_path (str, optional): path to the actor weights. Loads the
                weights is the path is not None. Defaults to None.
            critic_path (str, optional):  path to the critic weights. Loads the
                weights is the path is not None. . Defaults to None.
        """
        super().build_network(
            verbose, actor_path=actor_path, critic_path=critic_path,
            tie_conv=False
        )

        # Disable actor gradient
        eval_only(self.actor)

    def update_actor(self, batch) -> Tuple[float, float]:
        """
        Updates the latent distribution (mean and std) given the sampled
        transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of PAC-Bayes bound regularization.
        """
        (_, _, state, _, _, _, _, append, _, _, _, _, _) = batch
        self.critic.eval()
        self.actor.eval()

        # Re-sample latent using current latent distribution
        latent = self.latent_ps_copy.rsample((len(state),)).to(self.device)
        action_sample = self.actor(
            state, append=append, latent=latent, detach_encoder=False
        )
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(
            state, action_sample, append=append, latent=latent,
            detach_encoder=False
        )

        # conservative estimation
        if self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)
        else:
            q_pi = torch.max(q_pi_1, q_pi_2)

        loss_q_eval = q_pi.mean()
        loss_pi = -loss_q_eval

        if self.bound_type == 'kl':
            div = self.get_kl_div_estimate(latent)
        elif self.bound_type == 'renyi':
            div = self.get_renyi_div()

        # do not backpropagate if infinite renyi
        with torch.no_grad():
            if torch.any(torch.isinf(self.get_renyi_div())):
                div = 0
        loss_bound_reg = self.div_weight * div
        loss_pi += loss_bound_reg

        self.latent_optimizer.zero_grad()
        loss_pi.backward()
        self.latent_optimizer.step()

        # Project logvar
        # with torch.no_grad():
        #     self.clip_logvar()

        # # Automatic temperature tuning
        # loss_alpha = (self.alpha *
        #               (-log_prob - self.target_entropy).detach()).mean()
        # if self.learn_alpha:
        #     self.log_alpha_optimizer.zero_grad()
        #     loss_alpha.backward()
        #     self.log_alpha_optimizer.step()
        # return loss_pi.item(), loss_entropy.item(), loss_alpha.item()
        if torch.is_tensor(loss_bound_reg):
            loss_bound_reg = loss_bound_reg.item()
        return loss_pi.item(), loss_bound_reg

    def update_latent(self):
        """Updates the latent distribution.
        """
        self.latent_mean = self.latent_mean_copy.clone().detach()
        self.latent_base_logvar = self.latent_base_logvar_copy.clone().detach()
        self.latent_mean.requires_grad = False
        self.latent_base_logvar.requires_grad = False
        logvar = baselogvar2logvar(
            self.latent_base_logvar, self.latent_prior_std[0]
        )
        self.latent_std = logvar2std(logvar)
        self.latent_ps = torch.distributions.Normal(
            self.latent_mean, self.latent_std
        )
        self.latent_optimizer_state = self.latent_optimizer.state_dict()

        del self.latent_mean_copy
        del self.latent_base_logvar_copy
        del self.latent_std_copy
        del self.latent_ps_copy
        del self.latent_optimizer

    def update_actor_hyper_param(self):  # Overrides, actor weights are fixed.
        pass

    def get_kl_div(self):
        """Gets the KL divergence between the posterior and the prior.

        Returns:
            float: KL divergence between the posterior and the prior.
        """
        return torch.distributions.kl.kl_divergence(
            self.latent_ps, self.latent_prior
        ).sum()

    def get_kl_div_estimate(self, latent: torch.Tensor):
        """Gets the difference between log probabilities given a latent.

        Returns:
            float: sampled KL divergence.
        """
        return torch.mean(
            torch.sum(
                self.latent_ps.log_prob(latent)
                - self.latent_prior.log_prob(latent), dim=1
            )
        )

    def get_renyi_div(self, a: float = 2.) -> float:
        """Gets the Renyi divergence between the posterior and the prior.

        Args:
            a (float, optional): the order of Renyi divergence. Defaults to 2.

        Returns:
            float: Renyi divergence between the posterior and the prior.
        """
        mu1 = self.latent_mean
        logvar1 = std2logvar(self.latent_std)
        mu2 = self.latent_prior_mean
        logvar2 = std2logvar(self.latent_prior_std)

        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        vara = a*var2 + (1-a) * var1
        finiteness_check = a * (1/var1) + (1-a) * (1/var2)
        if torch.sum(finiteness_check > 0) < mu1.shape[0]:
            return torch.Tensor([float("Inf")])
        sum_logvara = torch.sum(torch.log(vara))
        sum_logvar1 = torch.sum(logvar1)
        sum_logvar2 = torch.sum(logvar2)

        r_div = (a/2) * torch.sum(((mu1 - mu2)**2) * vara)
        r_div -= 1 / (2*a -
                      2) * (sum_logvara - (1-a) * sum_logvar1 - a*sum_logvar2)
        return r_div

    def clip_logvar(self, ps_std_max_multiplier: float = 2):
        """Clips the log variance of the posterior.

        Args:
            ps_std_max_multiplier (float, optional): the maximum multiplier to
                the standard deviation of the prior. Defaults to 2.
        """
        eps = 1e-6
        ps_std_max_multiplier = torch.tensor(ps_std_max_multiplier * 1.0)
        max_logvar = (
            std2logvar(self.latent_prior_std)
            + torch.log(ps_std_max_multiplier) - eps
        )
        logvar = baselogvar2logvar(
            self.latent_base_logvar_copy, self.latent_prior_std[0]
        )
        projected_logvar = torch.min(logvar, max_logvar)
        self.latent_base_logvar_copy.copy_(
            logvar2baselogvar(projected_logvar, self.latent_prior_std[0])
        )

    def save(self, step: int, logs_path: str):
        """Saves the latent distribution.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
        """
        os.makedirs(os.path.join(logs_path, 'latent'), exist_ok=True)
        latent_dict = {
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std
        }
        path_latent = os.path.join(
            logs_path, 'latent', '{}-{}.pth'.format('latent', step)
        )
        torch.save(latent_dict, path_latent)
        print('=> Save {} after [{}] updates'.format(path_latent, step))

    def remove(self, step: int, logs_path: str):
        """Removes the latent distribution checkpoints.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
        """
        path_latent = os.path.join(
            logs_path, 'latent', 'latent-{}.pth'.format(step)
        )
        print("Remove", path_latent)
        if os.path.exists(path_latent):
            os.remove(path_latent)
