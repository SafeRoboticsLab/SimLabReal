# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for soft actor-critic variants without latent variables.

This file implements a parent class for all soft actor-critic (SAC) variants
without latent variables in this project. A child class should implement the
following functions
    + update / update_actor / update_critic: customized functions to update the
        actor or the critic.
"""

import torch
from torch.nn.functional import mse_loss

from agent.base_SAC import BaseSAC


class SAC_mini(BaseSAC):

    def __init__(self, cfg, cfg_arch, cfg_env):
        super().__init__(cfg, cfg_arch, cfg_env)

    @property
    def has_latent(self):
        return False

    @property
    def latent_dist(self):
        return None

    def build_network(
        self, build_optimizer=True, verbose: bool = True, actor_path=None,
        critic_path=None, tie_conv=True
    ):
        """Builds neural networks for critic and actor(s).

        Args:
            build_optimizer (bool): builds optimizers for the neural networks
                if Ture. Defaults to True.
            verbose (bool): prints out meassages if Ture. Defaults to True.
            actor_path (str, optional): path to the actor weights. Loads the
                weights is the path is not None. Defaults to None.
            critic_path (str, optional):  path to the critic weights. Loads the
                weights is the path is not None. Defaults to None.
            tie_conv (bool, optional): ties the actor's encoder with the
                critic's encoder if True. Defaults to True.
        """
        super().build_network(
            verbose, actor_path=actor_path, critic_path=critic_path,
            tie_conv=tie_conv
        )

        # Sets up optimizer
        if build_optimizer:
            super().build_optimizer()
        else:
            for _, param in self.actor.named_parameters():
                param.requires_grad = False
            for _, param in self.critic.named_parameters():
                param.requires_grad = False
            self.actor.eval()
            self.critic.eval()

    # main update functions
    def update_critic(self, batch):
        """Updates the critic given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of critic from the Bellman equation.
        """
        (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x, _,
            append, non_final_append_nxt, l_x_ra, binary_cost, _, _
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

        # == compute actor next_actions and feed to critic_target ==
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                non_final_state_nxt, append=non_final_append_nxt
            )
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
                next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.critic_target(
                non_final_state_nxt, next_actions, append=non_final_append_nxt
            )
            # max for reach-avoid Bellman equation, safety Bellman equation and
            # risk (recovery RL)
            if (
                self.mode == 'RA' or self.mode == 'safety'
                or self.mode == 'risk'
            ):
                q_max = torch.max(next_q1, next_q2).view(-1)
            elif self.mode == 'performance':
                q_min = torch.min(next_q1, next_q2).view(-1)
            else:
                raise ValueError("Unsupported RL mode.")

            final_mask = torch.logical_not(non_final_mask)
            if self.mode == 'RA':
                y[non_final_mask] = (
                    (1.0 - self.gamma)
                    * torch.max(l_x_ra[non_final_mask], g_x[non_final_mask])
                    + self.gamma * torch.max(
                        g_x[non_final_mask],
                        torch.min(l_x_ra[non_final_mask], q_max)
                    )
                )
                if self.terminal_type == 'g':
                    y[final_mask] = g_x[final_mask]
                elif self.terminal_type == 'max':
                    y[final_mask] = torch.max(
                        l_x_ra[final_mask], g_x[final_mask]
                    )
                else:
                    raise ValueError("invalid terminal type")
            elif self.mode == 'safety':
                # V(s) = max{ g(s), V(s') }
                # Q(s, u) = V( f(s,u) ) = max{ g(s'), min_{u'} Q(s', u') }
                # normal state
                y[non_final_mask] = ((1.0 - self.gamma) * g_x[non_final_mask]
                                     + self.gamma
                                     * torch.max(g_x[non_final_mask], q_max))

                # terminal state
                y[final_mask] = g_x[final_mask]
            elif self.mode == 'performance':
                target_q = q_min - self.alpha * next_log_prob.view(
                    -1
                )  # already masked - can be lower dim than y
                y = reward
                y[non_final_mask] += self.gamma * target_q
            elif self.mode == 'risk':
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

    def update_actor(self, batch):
        """Updates the actor given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
            float: loss of entropy coefficient (alpha).
        """
        _, _, state, _, _, _, _, append, _, _, _, _, _ = batch

        self.critic.eval()
        self.actor.train()

        action_sample, log_prob = self.actor.sample(
            state, append=append, detach_encoder=True
        )  # Uses detach_encoder=True to not update conv layers.
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(
            state, action_sample, append=append, detach_encoder=True
        )

        if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)

        # cost: min_theta E[ Q + alpha * (log pi + H)]
        # loss_pi = Q + alpha * log pi
        # reward: max_theta E[ Q - alpha * (log pi + H)]
        # loss_pi = -Q + alpha * log pi
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
            loss_pi = loss_q_eval + self.alpha * loss_entropy
        elif self.mode == 'performance':
            loss_pi = -loss_q_eval + self.alpha * loss_entropy
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

    def update(self, batch, timer, update_period=2):
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
        """
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha

    def value(self, obs, append):
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
