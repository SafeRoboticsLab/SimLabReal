# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""A class for soft actor-critic with diversity-motivated reward.

This file implements an augmented reward motivated by maximizing the mutual
information (MI) between the state and the latent and minimizing the MI between
the action and the latent given the state. This augemented reward is derived
from ``Diversity is All You Need: Learning Skills without a Reward Function'',
which we refer to observation-marginal MI in our paper.

Paper: https://arxiv.org/abs/1802.06070.
"""

import os
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam

from agent.SAC_latent import SAC_latent
from agent.model import Discriminator
from agent.scheduler import StepLRFixed
from utils.misc import save_model, optimizer_to


class SAC_maxEnt(SAC_latent):
    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            config (Class object): training hyper-parameters.
            config_arch (Class object): NN architecture hyper-parameters.
            config_env (Class object): environment hyper-parameters.
        """
        super().__init__(cfg, cfg_arch, cfg_env)

        # Discriminator training
        self.lr_d = cfg.lr_d
        log_ps_max = 2
        log_ps_min = -2

        # Aug reward for maxEnt
        if cfg.aug_reward_range_schedule:
            self.aug_reward_range_scheduler = StepLRFixed(
                init_value=cfg.aug_reward_range_init,
                period=cfg.aug_reward_range_period,
                end_value=cfg.aug_reward_range_end,
                step_size=cfg.aug_reward_range_step,
            )
            self.update_aug_reward_range()
        else:
            self.aug_reward_range = cfg.aug_reward_range
            self.aug_reward_range_scheduler = None

        # Latent mean
        self.latent_mean = torch.zeros((self.latent_dim)).to(self.device)

        # Latent std
        self.log_ps_bound_upper = self.latent_dim * log_ps_max
        self.log_ps_bound_lower = self.latent_dim * log_ps_min
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
                (self.latent_dim)).to(self.device)  # unit gaussian
            self.latent_prior = torch.distributions.Normal(
                self.latent_mean, self.latent_std)

    @property
    def latent_dist(self):
        return self.latent_prior

    def build_network(self, build_optimizer=True, verbose=True):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to True.
        """
        super().build_network(verbose)
        self.disc = Discriminator(
            input_n_channel=self.obs_channel,
            latent_dim=self.latent_dim,
            mlp_dim=self.cfg_arch.mlp_dim.disc,
            append_dim=self.cfg_arch.append_dim,
            img_sz=[self.img_h, self.img_w],
            kernel_sz=self.cfg_arch.kernel_size,
            stride=self.cfg_arch.stride,
            n_channel=self.cfg_arch.n_channel,
            use_spec=self.cfg_arch.use_spec_disc,
            use_sm=False,
            use_ln=False,
            device=self.device,
            verbose=verbose,
        )

        # Set up optimizer
        if build_optimizer:
            self.build_optimizer()
        else:
            for _, param in self.actor.named_parameters():
                param.requires_grad = False
            for _, param in self.critic.named_parameters():
                param.requires_grad = False
            for _, param in self.disc.named_parameters():
                param.requires_grad = False
            self.actor.eval()
            self.critic.eval()
            self.disc.eval()

    def build_optimizer(self):
        """Builds optimizers for NNs and schedulers for learning rates.
        """
        super().build_optimizer()
        self.disc_optimizer = Adam(self.disc.parameters(), lr=self.lr_d)

    # main update functions
    def update_aug_reward_range(self):
        """Updates the range of augmented reward by stepping the scheduler.
        """
        self.aug_reward_range = self.aug_reward_range_scheduler.get_variable()

    def update_latent(self):
        """Updates the latent distribution by stepping its std scheduler.
        """
        self.latent_prior_std = self.latent_std_scheduler.get_variable()
        self.latent_std = self.latent_prior_std * torch.ones(
            (self.latent_dim)).to(self.device)  # unit gaussian
        self.latent_prior = torch.distributions.Normal(self.latent_mean,
                                                       self.latent_std)

    def update_critic(self, batch):
        """Updates the critic given the sampled transitions from the buffer.

        This update includes an augmented reward motivated by maximizing the
        mutual information between the state and the latent and minimizing the
        mutual information between the action and the latent given the state.
        This augemented reward is derived from ``Diversity is All You Need:
        Learning Skills without a Reward Function''.

        Paper: https://arxiv.org/abs/1802.06070.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of critic from the Bellman equation.
        """
        (non_final_mask, non_final_state_nxt, state, action, reward, g_x,
         latent, append, non_final_append_nxt, l_x_ra, _, _, _) = batch
        self.critic.train()
        self.critic_target.eval()
        self.actor.eval()
        self.disc.eval()

        # Augments reward with entropy for non final state
        with torch.no_grad():
            non_final_latent = latent[non_final_mask]
            # non_final_state_stacked = torch.cat((state[non_final_mask],
            # non_final_state_nxt), dim=1)
            # log_prob_ps = self.disc(
            #     non_final_state_stacked, non_final_latent
            # ).sum(dim=1)
            log_prob_ps = self.disc(non_final_state_nxt,
                                    non_final_append_nxt,
                                    latent=non_final_latent).sum(dim=1)
            log_prob_prior = self.latent_prior.log_prob(non_final_latent).sum(
                dim=1)
            aug_cost = 0.5 - (log_prob_ps - log_prob_prior).clamp(
                min=self.log_ps_bound_lower, max=self.log_ps_bound_upper) / (
                    self.log_ps_bound_upper - self.log_ps_bound_lower
                )  # normalizes by centering at [-0.5,0.5] instead of [0,1]
            reward[non_final_mask] -= self.aug_reward_range * aug_cost

        # Gets Q(s,a)
        if not self.critic_has_act_ind:
            action = action[:, :self.action_dim]
        q1, q2 = self.critic(state, action, append=append, latent=latent)

        # == placeholder for target ==
        y = torch.zeros(self.batch_size).float().to(self.device)

        # Computes next_actions from the actor and feeds to critic_target
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                non_final_state_nxt,
                append=non_final_append_nxt,
                latent=non_final_latent)
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
                next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.critic_target(non_final_state_nxt,
                                                  next_actions,
                                                  append=non_final_append_nxt,
                                                  latent=non_final_latent)
            if self.mode == 'RA' or self.mode == 'safety':
                q_max = torch.max(next_q1, next_q2).view(-1)
            elif self.mode == 'performance':
                q_min = torch.min(next_q1, next_q2).view(-1)
            else:
                raise ValueError("Unsupported RL mode.")

            final_mask = torch.logical_not(non_final_mask)
            if self.mode == 'RA':
                y[non_final_mask] = (
                    (1.0 - self.gamma) *
                    torch.max(l_x_ra[non_final_mask], g_x[non_final_mask]) +
                    self.gamma *
                    torch.max(g_x[non_final_mask],
                              torch.min(l_x_ra[non_final_mask], q_max)))
                if self.terminal_type == 'g':
                    y[final_mask] = g_x[final_mask]
                elif self.terminal_type == 'max':
                    y[final_mask] = torch.max(l_x_ra[final_mask],
                                              g_x[final_mask])
                else:
                    raise ValueError("invalid terminal type")
            elif self.mode == 'safety':
                # V(s) = max{ g(s), V(s') }, Q(s, u) = V( f(s,u) )
                # normal state
                g_x_non_final = g_x[non_final_mask]
                y[non_final_mask] = (
                    (1.0 - self.gamma) * g_x_non_final +
                    self.gamma * torch.max(g_x_non_final, q_max))
                # y[non_final_mask] += self.aug_reward_range * aug_cost

                # terminal state
                y[final_mask] = g_x[final_mask]
            elif self.mode == 'performance':
                target_q = q_min - self.alpha * next_log_prob.view(
                    -1)  # already masked - can be lower dim than y
                y = reward
                y[non_final_mask] += self.gamma * target_q

        # Uses MSE to update for both Q1 and Q2
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        # Backpropagates
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        return loss_q.item()

    def update_disc(self, batch):
        """
        Updates the latent distribution given the sampled transitions from the
        buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of latent distribution.
        """
        (non_final_mask, non_final_state_nxt, _, _, _, _, latent, _,
         non_final_append_nxt, _, _, _, _) = batch
        self.disc.train()
        # non_final_state_stacked = torch.cat((state[non_final_mask],
        # non_final_state_nxt), dim=1)
        # log_prob_ps = self.disc(
        #     non_final_state_stacked, latent[non_final_mask],
        #     detach_encoder=False
        # ).sum(dim=1)  # not sharing weights for disc
        log_prob_ps = self.disc(non_final_state_nxt,
                                non_final_append_nxt,
                                latent[non_final_mask],
                                detach_encoder=False).sum(
                                    dim=1)  # not sharing weights for disc
        loss_disc = -log_prob_ps.mean()

        # == backpropagation ==
        self.disc_optimizer.zero_grad()
        loss_disc.backward()
        self.disc_optimizer.step()
        return loss_disc.item()

    def update_actor(self, batch):
        """Updates the actor given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of entropy regularization.
            float: loss of entropy coefficient (alpha).
        """
        (_, _, state, _, _, _, latent, append, _, _, _, _, _) = batch
        self.critic.eval()
        self.actor.train()
        self.disc.eval()

        action_sample, log_prob = self.actor.sample(state,
                                                    append=append,
                                                    latent=latent,
                                                    detach_encoder=True)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(state,
                                     action_sample,
                                     append=append,
                                     latent=latent,
                                     detach_encoder=True)

        if self.mode == 'RA' or self.mode == 'safety':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)

        # Obj: min_theta E[ Q(s, pi_theta(s)) + alpha * log(pi_theta(s))]
        # loss_pi = (q_pi + self.alpha * log_prob.view(-1)).mean()
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode == 'safety':
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
        # == EXPERIENCE REPLAY ==
        self.critic.train()
        self.actor.train()

        # Updates critic
        loss_q = self.update_critic(batch)
        # Updates actor and target critic
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha

    def save(self, step, logs_path, max_model=None):
        """Saves the current actor, critic, and the latent distribution.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
            max_model (int, optional): the maximum number of models to save.
                Defaults to None.
        """
        super().save(step, logs_path)
        path_d = os.path.join(logs_path, 'disc')
        save_model(self.disc, step, path_d, 'disc', max_model)

    def remove(self, step: int, logs_path: str):
        """Removes the actor, critic, and the latent distribution checkpoints.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
        """
        super().remove(step, logs_path)
        path_d = os.path.join(logs_path, 'disc', 'disc-{}.pth'.format(step))
        print("Remove", path_d)
        if os.path.exists(path_d):
            os.remove(path_d)

    def move_all_to_cpu(self):
        super().move_all_to_cpu()
        self.disc.to('cpu')
        self.disc_optimizer.zero_grad()
        optimizer_to(self.disc_optimizer, 'cpu')
        self.latent_prior.loc = self.latent_prior.loc.cpu()
        self.latent_prior.scale = self.latent_prior.scale.cpu()

    def move_all_to_prev_device(self):
        super().move_all_to_prev_device()
        self.disc.to(self.device)
        optimizer_to(self.disc_optimizer, self.device)
        self.latent_prior.loc = self.latent_prior.loc.to(self.device)
        self.latent_prior.scale = self.latent_prior.scale.to(self.device)
