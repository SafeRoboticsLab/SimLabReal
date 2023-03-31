# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic with diversity-motivated reward.

This file implements an augmented reward motivated by maximizing the mutual
information (MI) between the action and the latent given the state. This
augemented reward is derived from ``Dynamics-Aware Unsupervised Discovery of
Skills'', which we refer to observation-conditional MI in our paper.

Paper: https://arxiv.org/abs/1907.01657.
"""

from typing import Tuple
import torch

from agent.SAC_latent import SAC_latent
from agent.scheduler import StepLRFixed


class SAC_dads_prior(SAC_latent):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            cfg (Class object): training hyper-parameters.
            cfg_arch (Class object): NN architecture hyper-parameters.
            cfg_env (Class object): environment hyper-parameters.
        """
        super().__init__(cfg, cfg_arch, cfg_env)
        assert self.learn_alpha is False, (
            "currently not supporting alpha tuning."
        )
        self.g_x_MI = cfg_env.g_x_MI

        # Latent
        self.num_latent_samples = cfg.num_latent_samples
        self.latent_mean = torch.zeros((self.latent_dim)).to(self.device)

        if cfg.latent_std_schedule:
            self.latent_std_scheduler = StepLRFixed(
                init_value=cfg.latent_std_init,
                period=cfg.latent_std_period,
                end_value=cfg.latent_std_end,
                step_size=cfg.latent_std_step,
                min_step=cfg.latent_std_min_step,
            )
            self.update_latent()
        else:
            self.latent_prior_std = cfg.latent_prior_std
            self.latent_std = self.latent_prior_std * torch.ones(
                (self.latent_dim)
            ).to(self.device)  # unit gaussian
            self.latent_prior = torch.distributions.Normal(
                self.latent_mean, self.latent_std
            )

        if cfg.alpha_schedule:
            self.alpha_scheduler = StepLRFixed(
                init_value=cfg.alpha_init,
                period=cfg.alpha_period,
                end_value=cfg.alpha_end,
                step_size=cfg.alpha_step,
                min_step=cfg.alpha_step,
            )
            self.update_alpha()
        else:
            self.dads_alpha = cfg.alpha

    def update_latent(self):
        """Updates the latent distribution by stepping its std scheduler.
        """
        self.latent_prior_std = self.latent_std_scheduler.get_variable()
        self.latent_std = self.latent_prior_std * torch.ones(
            (self.latent_dim)
        ).to(self.device)  # unit gaussian
        self.latent_prior = torch.distributions.Normal(
            self.latent_mean, self.latent_std
        )

    def update_alpha(self):
        """Updates the learning rate of alpha.
        """
        self.dads_alpha = self.alpha_scheduler.get_variable()

    @property
    def latent_dist(self):
        return self.latent_prior

    def build_network(self, verbose: bool = True):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to True.
        """
        super().build_network(verbose)

        # Sets up optimizer
        super().build_optimizer()

    # main update functions
    def update_actor(self, batch) -> Tuple[float, float]:
        """Updates the actor given the sampled transitions from the buffer.

        This update includes a mutual information term, I(A; Z | S), that we
        want to maximize. It approximates p(a | s) by sampling z's from the
        latent distribution. To be more specific, p(a | s) = sum_{i=1}^L
        pi(a | s, z) / L.

        Args:
            batch (tuple).

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
        """
        (_, _, state, action, _, _, latent, append, _, _, _, _, _) = batch
        self.critic.eval()
        self.actor.train()

        with torch.no_grad():
            num_sample = int(state.shape[0] * self.num_latent_samples)
            latent_samples = self.latent_prior.sample((num_sample,)
                                                     ).to(self.device)

        # Re-sample action for MI
        action_sample, log_prob, log_marginal = self.actor.sample_and_MI(
            state, append=append, latent=latent, detach_encoder=True,
            latent_samples=latent_samples
        )
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

        # Use action from batch
        # action_sample, log_prob = self.actor.sample(
        #     state, append=append, latent=latent, detach_encoder=True
        # )
        # if self.critic_has_act_ind:
        #     action = action[:, :-1]
        # log_marginal = self.actor.get_log_marginal(
        #     state, append, action, latent_samples, detach_encoder=True
        # )

        q_pi_1, q_pi_2 = self.critic(
            state, action_sample, append=append, latent=latent,
            detach_encoder=True
        )

        if self.mode == 'RA' or self.mode == 'safety':
            q_pi = torch.max(q_pi_1, q_pi_2)
            mask = q_pi > self.g_x_MI
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)
            mask = torch.ones_like(log_marginal, dtype=bool)

        # Obj: min_theta E[ Q(s, pi_theta(s)) - I(A; Z | S) ]
        # loss_pi = q_pi + alpha*(log p(a|s) - log pi(a|s,z))
        loss_MI = (log_marginal - log_prob)[mask].view(-1).mean()
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode == 'safety':
            loss_pi = loss_q_eval + self.dads_alpha * loss_MI
        elif self.mode == 'performance':
            loss_pi = -loss_q_eval + self.dads_alpha * loss_MI
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        return loss_pi.item(), loss_MI.item()

    def update(self, batch, timer: int,
               update_period: int = 2) -> Tuple[float, float, float]:
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
        """
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi = 0.
        loss_MI = 0.
        if timer % update_period == 0:
            loss_pi, loss_MI = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_MI
