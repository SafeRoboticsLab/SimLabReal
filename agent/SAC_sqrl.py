# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for safety critic learning of pre-training phase in SQRL paper.

This file implements safety critic learning for pre-training phase in
``Learning to be Safe: Deep RL with a Safety Critic'' (SQRL).

Paper: https://arxiv.org/abs/2010.14603.
"""

from typing import Tuple
import torch
from torch.nn.functional import mse_loss

from agent.SAC_mini import SAC_mini


class SAC_sqrl(SAC_mini):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            CONFIG (object): update-rekated hyper-parameter configuration.
            cfg_arch (object): NN architecture configuration.
            cfg_env (object): environment configuration.
        """
        super(SAC_sqrl, self).__init__(cfg, cfg_arch, cfg_env)
        assert self.mode == 'risk', "SQRL only uses risk critic."

    def update_critic(self, batch) -> float:
        """Updates the critic given the sampled transitions from the buffer.

        Implements a SARSA-like update, which uses action_nxt stored in the
        buffer. Note that action_next should include action indicators (this
        action comes from the safety policy or performance policy.).

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of critic from the Bellman equation.
        """
        (
            non_final_mask, non_final_state_nxt, state, action, _, _, _,
            append, non_final_append_nxt, _, binary_cost, _, _, action_next
        ) = batch
        self.critic.train()
        self.critic_target.eval()
        self.actor.eval()

        # == get Q(s,a) ==
        q1, q2 = self.critic(
            state, action, append=append
        )  # Used to compute loss (non-target part).

        # == placeholder for target ==
        y = torch.zeros(self.batch_size).float().to(self.device)

        # == use action_next from the batch and feed to critic_target ==
        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(
                non_final_state_nxt, action_next, append=non_final_append_nxt
            )
            q_max = torch.max(next_q1, next_q2).view(-1)
            y = binary_cost  # y = 1 if it's a terminal state
            y[non_final_mask] += self.gamma * q_max

        # == MSE update for both Q1 and Q2 ==
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        # == backpropagation ==
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        return loss_q.item()

    def update_actor(self, batch) -> Tuple[float, float, float]:
        """Updates the actor given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
            float: loss of entropy coefficient (alpha).
        """
        _, _, state, _, _, _, _, append, _, _, _, _, _, _ = batch

        self.critic.eval()
        self.actor.train()

        action_sample, log_prob = self.actor.sample(
            state, append=append, detach_encoder=True
        )
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(
            state, action_sample, append=append, detach_encoder=True
        )

        # It minimizes the risk, so it uses max to have conservative
        # estimations from twinned Q-network.
        q_pi = torch.max(q_pi_1, q_pi_2)

        # cost: min_theta E[ Q + alpha * (log pi + H)]
        # loss_pi = Q + alpha * log pi
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        loss_pi = loss_q_eval + self.alpha * loss_entropy
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
        return loss_pi.item(), loss_entropy.item(), loss_alpha.item()
