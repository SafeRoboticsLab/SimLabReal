# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
This file implements a SAC variant with a trainable latent distribution. The
weights of actor and critic networks are fixed.
"""

from typing import Optional, Tuple
from agent.SAC_ps import SAC_ps
from utils.misc import eval_only


class SAC_ps_noc(SAC_ps):

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

        # Disable critic gradient
        eval_only(self.critic)

    def update(self, batch, timer: int,
               update_period: int = 2) -> Tuple[float, float]:
        """
        Updates the latent distribution given the sampled transitions from the
        buffer.

        Args:
            batch (tuple): sampled transitions from the buffer.
            timer (int): the timer of the number of updates happened in a
                single optimization.
            update_period (int): the period to update the actor and the critic
                target.

        Returns:
            float: loss of actor indicated by the critic.
            float: loss of PAC-Bayes bound regularization.
        """
        # Updatess latent.
        loss_pi, loss_bound_reg = 0, 0
        if timer % update_period == 0:
            self.build_latent_optimzer()
            loss_pi, loss_bound_reg = self.update_actor(batch)
            self.update_latent()

        return loss_pi, loss_bound_reg

    def build_optimizer(self):  # Overrides, actor/critic weights are fixed.
        pass

    def update_critic(self, batch):  # Overrides, critic weights are fixed.
        pass

    def update_hyper_param(self):  # Overrides, actor/critic weights are fixed.
        pass

    def update_target_networks(self):  # Overrides, critic weights are fixed.
        pass

    def save(self, **kwargs):  # Overrides, actor/critic weights are fixed.
        pass

    def remove(self, **kwargs):  # Overrides, actor/critic weights are fixed.
        pass
