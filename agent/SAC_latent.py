# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for soft actor-critic variants without latent variables.

This file implements a parent class for all soft actor-critic (SAC) variants
conditioned on latent variables in this project. Specifically, this class
serves as a parent class for SAC_maxent, SAC_dads and SAC_ps. A child class
should implement the following functions
    + latent_dist: the distribution of the latent variables.
    + update / update_actor / update_critic: customized functions to update the
        actor or the critic.
"""

import torch
from torch.nn.functional import mse_loss

from agent.base_SAC import BaseSAC


class SAC_latent(BaseSAC):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            cfg (Class object): training hyper-parameters.
            cfg_arch (Class object): NN architecture hyper-parameters.
            cfg_env (Class object): environment hyper-parameters.
        """
        super().__init__(cfg, cfg_arch, cfg_env)

        # Latent
        self.latent_dim = cfg.latent_dim
        self.aug_reward_range_scheduler = None
        self.latent_std_scheduler = None
        self.alpha_scheduler = None

    @property
    def has_latent(self):
        return True

    # main update functions
    def update_critic(self, batch):
        """Updates the critic given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of critic from the Bellman equation.
        """
        (
            non_final_mask, non_final_state_nxt, state, action, reward, g_x,
            latent, append, non_final_append_nxt, l_x_ra, _, _, _
        ) = batch
        self.critic.train()
        self.critic_target.eval()
        self.actor.eval()
        non_final_latent = latent[non_final_mask]

        # == get Q(s,a) ==
        q1, q2 = self.critic(
            state, action, append=append, latent=latent
        )  # Used to compute loss (non-target part).

        # == placeholder for target ==
        y = torch.zeros(self.batch_size).float().to(self.device)

        # == compute actor next_actions and feed to critic_target ==
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                non_final_state_nxt, append=non_final_append_nxt,
                latent=non_final_latent
            )
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
                next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.critic_target(
                non_final_state_nxt, next_actions, append=non_final_append_nxt,
                latent=non_final_latent
            )
            # use max for RA or safety
            if self.mode == 'RA' or self.mode == 'safety':
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
                # V(s) = max{ g(s), V(s') }, Q(s, u) = V( f(s,u) )
                # normal state
                non_final_g_x = g_x[non_final_mask]
                y[non_final_mask] = ((1.0 - self.gamma) * non_final_g_x
                                     + self.gamma
                                     * torch.max(non_final_g_x, q_max))

                # terminal state
                y[final_mask] = g_x[final_mask]
            elif self.mode == 'performance':
                target_q = q_min - self.alpha * next_log_prob.view(
                    -1
                )  # already masked - can be lower dim than y
                y = reward
                y[non_final_mask] += self.gamma * target_q

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
        """
        (_, _, state, _, _, _, latent, append, _, _, _, _, _) = batch
        self.critic.eval()
        self.actor.train()

        action_sample, _ = self.actor.sample(
            state, append=append, latent=latent, detach_encoder=True
        )
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)

        q_pi_1, q_pi_2 = self.critic(
            state, action_sample, append=append, latent=latent,
            detach_encoder=True
        )

        if self.mode == 'RA' or self.mode == 'safety':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)

        # Obj: min_theta E[ Q(s, pi_theta(s)) ]
        # loss_pi = q_pi
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode == 'safety':
            loss_pi = loss_q_eval
        elif self.mode == 'performance':
            loss_pi = -loss_q_eval
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        return loss_pi.item()

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
        """
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi = 0.
        if timer % update_period == 0:
            loss_pi = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi

    def value(self, obs, append):
        """Returns the critic evaluation at this observation and append info.

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

    def move_all_to_cpu(self):
        super().move_all_to_cpu()
        self.latent_mean = self.latent_mean.to('cpu')
        self.latent_std = self.latent_std.to('cpu')

    def move_all_to_prev_device(self):
        super().move_all_to_prev_device()
        self.latent_mean = self.latent_mean.to(self.device)
        self.latent_std = self.latent_std.to(self.device)
