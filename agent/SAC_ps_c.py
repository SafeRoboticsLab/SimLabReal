# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
This file implements a SAC variant with a trainable latent distribution and
critic. The weights of actor network is fixed.
"""

from typing import Optional, Tuple
import os
import copy
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim import lr_scheduler

from agent.SAC_ps import SAC_ps
from utils.misc import save_model


class SAC_ps_c(SAC_ps):

    def __init__(self, cfg, cfg_arch, cfg_env):
        """
        Args:
            cfg (Class object): training hyper-parameters.
            cfg_arch (Class object): NN architecture hyper-parameters.
            cfg_env (Class object): environment hyper-parameters.
        """
        super().__init__(cfg, cfg_arch, cfg_env)

    def build_network(
        self, verbose: bool = True, actor_path: Optional[str] = None,
        critic_path: Optional[str] = None
    ):
        """Builds neural networks for critic and actor(s).

        Args:
            verbose (bool): prints out meassages if Ture. Defaults to True.
            actor_path (str, optional): path to the actor weights. Loads the
                weights is the path is not None. Defaults to None.
            critic_path (str, optional):  path to the critic weights. Loads the
                weights is the path is not None. Defaults to None.
        """
        super().build_network(
            verbose, actor_path=actor_path, critic_path=critic_path
        )

        # Copy critic target
        self.critic_target = copy.deepcopy(self.critic)

        # Sets up optimizer
        self.build_optimizer()

    def update(self, batch, timer: int,
               update_period: int = 2) -> Tuple[float, float, float]:
        """
        Updates the latent distribution and the critic given the sampled
        transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
            timer (int): the timer of the number of updates happened in a
                single optimization.
            update_period (int): the period to update the actor and the critic
                target.

        Returns:
            float: loss of critic from the Bellman equation.
            float: loss of actor indicated by the critic.
            float: loss of PAC-Bayes bound regularization.
        """
        self.critic.train()

        # Updatess critic.
        loss_q = self.update_critic(batch)

        # Updatess actor, latent, and target critic.
        loss_pi, loss_bound_reg = 0, 0
        if timer % update_period == 0:
            self.build_latent_optimzer()
            loss_pi, loss_bound_reg = self.update_actor(batch)
            self.update_target_networks()
            self.update_latent()

        self.critic.eval()

        return loss_q, loss_pi, loss_bound_reg

    def build_optimizer(self):
        """Builds optimizers for the critic and the latent distribution.
        """
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_c)

        if self.lr_c_schedule:
            self.critic_scheduler = lr_scheduler.StepLR(
                self.critic_optimizer, step_size=self.lr_c_period,
                gamma=self.lr_c_decay
            )

    def update_critic(self, batch):
        """Updates the critic given the sampled transitions from the buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.

        Returns:
            float: loss of critic from the Bellman equation.
        """
        (
            non_final_mask, non_final_state_nxt, state, action, reward, _, _,
            append, non_final_append_nxt, _, _, _, _
        ) = batch
        self.critic.train()
        self.critic_target.eval()

        #== Re-sample latent using current distribution ==
        with torch.no_grad():
            latent = self.latent_ps.sample((len(state),)).to(self.device)
            non_final_latent = latent[non_final_mask]

        #== get Q(s,a) ==
        if not self.critic_has_act_ind:
            action = action[:, :self.action_dim]
        q1, q2 = self.critic(
            state, action, append=append, latent=latent
        )  # Used to compute loss (non-target part).

        #== placeholder for target ==
        y = torch.zeros(self.batch_size).float().to(self.device)

        #== compute actor next_actions and feed to critic_target ==
        # only supports performance now
        with torch.no_grad():
            next_actions = self.actor(
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

            q_min = torch.min(next_q1, next_q2).view(-1)
            target_q = q_min
            y = reward
            y[non_final_mask] += self.gamma * target_q

        #== MSE update for both Q1 and Q2 ==
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        #== backpropagation ==
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        return loss_q.item()

    def update_hyper_param(self):
        """Updates all the hyper-parameters.
        """
        self.update_critic_hyper_param()

    def save(self, step: int, logs_path: str, max_model: Optional[int] = None):
        """Saves the current critic.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
            max_model (int, optional): the maximum number of models to save.
                Defaults to None.
        """
        super().save(step, logs_path)
        path_c = os.path.join(logs_path, 'critic')
        save_model(self.critic, step, path_c, 'critic', max_model)

    def remove(self, step: int, logs_path: str):
        """Removes the actor and critic checkpoints.

        Args:
            step (int): current number of training steps.
            logs_path (str): path to the model folder.
        """
        super().remove(step, logs_path)
        path_c = os.path.join(
            logs_path, 'critic', 'critic-{}.pth'.format(step)
        )
        print("Remove", path_c)
        if os.path.exists(path_c):
            os.remove(path_c)
