# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

# We train a performance policy safely given a backup policy trained in similar
# environment(s). Here we consider two shielding criteria: (1) using a forward
# simulator to check if from the new state we can remain safe within T_ro steps
# and (2) using safety critic values in every T_ch steps. Here the backup policy
# is a priori. The immediate next step is training the backup and performance
# policy together.

# We expect to reduce the number of collisions during the training without
# sacrifizing too much efficiency.

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import os
import pickle

from .model import SACPiNetwork, SACTwinnedQNetwork
from .scheduler import StepLRMargin
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model
import copy

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class policyShielding(object):
    def __init__(self, CONFIG, CONFIG_PERFORMANCE, CONFIG_BACKUP, obsChannel=3,
        verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        self.CONFIG = CONFIG
        self.CONFIG_PERFORMANCE = CONFIG_PERFORMANCE
        self.CONFIG_BACKUP = CONFIG_BACKUP
        self.saved = False
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== ENV PARAM ==
        self.obsChannel = obsChannel
        self.actionMag = CONFIG.ACTION_MAG
        self.actionDim = CONFIG.ACTION_DIM

        #== PARAM ==
        #= Learning
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        self.TAU = CONFIG.TAU  #= Target Network Update
        self.mode = CONFIG.MODE
        self.terminalType = CONFIG.TERMINAL_TYPE
        self.train_backup = CONFIG.TRAIN_BACKUP

        #= Discount Factor
        self.GammaScheduler = StepLRMargin( initValue=CONFIG.GAMMA,
            period=CONFIG.GAMMA_PERIOD, decay=CONFIG.GAMMA_DECAY,
            endValue=CONFIG.GAMMA_END, goalValue=1.)
        self.GAMMA = self.GammaScheduler.get_variable()
        self.GAMMA_PERIOD = CONFIG.GAMMA_PERIOD

        #= Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        self.LR_A = CONFIG.LR_A
        self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
        self.LR_A_DECAY = CONFIG.LR_A_DECAY
        self.LR_A_END = CONFIG.LR_A_END

        #= alpha-related hyper-parameters
        self.init_alpha = CONFIG.ALPHA
        self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.target_entropy = -self.actionDim
        self.LR_Al = CONFIG.LR_Al
        self.LR_Al_PERIOD = CONFIG.LR_Al_PERIOD
        self.LR_Al_DECAY = CONFIG.LR_Al_DECAY
        self.LR_Al_END = CONFIG.LR_Al_END
        if self.LEARN_ALPHA:
            print("SAC with learnable alpha and target entropy = {:.1e}".format(
                self.target_entropy))
        else:
            print("SAC with fixed alpha = {:.1e}".format(self.init_alpha))

        #= critic/actor-related hyper-parameters
        self.build_perforance_policy(CONFIG_PERFORMANCE, verbose=verbose)
        self.build_backup_policy(CONFIG_BACKUP, verbose=False)


    def build_perforance_policy(self, CONFIG, verbose=True):
        self.critic = SACTwinnedQNetwork(   input_n_channel=self.obsChannel,
                                            mlp_dim=CONFIG.MLP_DIM['critic'],
                                            actionDim=self.actionDim,
                                            actType=CONFIG.ACTIVATION['critic'],
                                            img_sz=CONFIG.IMG_SZ,
                                            kernel_sz=CONFIG.KERNEL_SIZE,
                                            n_channel=CONFIG.N_CHANNEL,
                                            use_sm=CONFIG.USE_SM,
                                            use_ln=CONFIG.USE_LN,
                                            device=self.device,
                                            verbose=verbose
        )
        self.criticTarget = copy.deepcopy(self.critic)

        if verbose:
            print("\nThe actor shares the same encoder with the critic.")
        self.actor = SACPiNetwork(  input_n_channel=self.obsChannel,
                                    mlp_dim=CONFIG.MLP_DIM['actor'],
                                    actionDim=self.actionDim,
                                    actionMag=self.actionMag,
                                    actType=CONFIG.ACTIVATION['actor'],
                                    img_sz=CONFIG.IMG_SZ,
                                    kernel_sz=CONFIG.KERNEL_SIZE,
                                    n_channel=CONFIG.N_CHANNEL,
                                    use_sm=CONFIG.USE_SM,
                                    use_ln=CONFIG.USE_LN,
                                    device=self.device,
                                    verbose=verbose
        )

        # Tie weights for conv layers
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # Set up optimizer
        self.build_optimizer()

        # Initialize alpha
        self.reset_alpha()


    def build_backup_policy(self, CONFIG, verbose=True):
        self.critic_backup = SACTwinnedQNetwork(   input_n_channel=self.obsChannel,
                                            mlp_dim=CONFIG.MLP_DIM['critic'],
                                            actionDim=self.actionDim,
                                            actType=CONFIG.ACTIVATION['critic'],
                                            img_sz=CONFIG.IMG_SZ,
                                            kernel_sz=CONFIG.KERNEL_SIZE,
                                            n_channel=CONFIG.N_CHANNEL,
                                            use_sm=CONFIG.USE_SM,
                                            use_ln=CONFIG.USE_LN,
                                            device=self.device,
                                            verbose=verbose
        )
        if self.train_backup:
            self.criticTarget_backup = copy.deepcopy(self.critic_backup)

        if verbose:
            print("\nThe actor shares the same encoder with the critic.")
        self.actor_backup = SACPiNetwork(  input_n_channel=self.obsChannel,
                                    mlp_dim=CONFIG.MLP_DIM['actor'],
                                    actionDim=self.actionDim,
                                    actionMag=self.actionMag,
                                    actType=CONFIG.ACTIVATION['actor'],
                                    img_sz=CONFIG.IMG_SZ,
                                    kernel_sz=CONFIG.KERNEL_SIZE,
                                    n_channel=CONFIG.N_CHANNEL,
                                    use_sm=CONFIG.USE_SM,
                                    use_ln=CONFIG.USE_LN,
                                    device=self.device,
                                    verbose=verbose
        )

        # Tie weights for conv layers
        self.actor_backup.encoder.copy_conv_weights_from(self.critic_backup.encoder)


    # * OTHERS STARTS
    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path, agentType):
        logs_path_critic = os.path.join(logs_path, 'critic')
        logs_path_actor = os.path.join(logs_path, 'actor')
        if agentType == 'backup':
            save_model(self.critic_backup, step, logs_path_critic, 'critic', self.MAX_MODEL)
            save_model(self.actor_backup,  step, logs_path_actor, 'actor',  self.MAX_MODEL)
        elif agentType == 'performance':
            save_model(self.critic, step, logs_path_critic, 'critic', self.MAX_MODEL)
            save_model(self.actor,  step, logs_path_actor, 'actor',  self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            config_path = os.path.join(logs_path, "CONFIG_PERFORMANCE.pkl")
            pickle.dump(self.CONFIG_PERFORMANCE, open(config_path, "wb"))
            config_path = os.path.join(logs_path, "CONFIG_BACKUP.pkl")
            pickle.dump(self.CONFIG_BACKUP, open(config_path, "wb"))
            self.saved = True


    def restore(self, step, logs_path, agentType):
        logs_path_critic = os.path.join(
            logs_path, 'model', 'critic', 'critic-{}.pth'.format(step))
        logs_path_actor  = os.path.join(
            logs_path, 'model', 'actor', 'actor-{}.pth'.format(step))
        if agentType == 'backup':
            self.critic_backup.load_state_dict(
                torch.load(logs_path_critic, map_location=self.device))
            self.critic_backup.to(self.device)
            if self.train_backup:
                self.criticTarget_backup.load_state_dict(
                    torch.load(logs_path_critic, map_location=self.device))
                self.criticTarget_backup.to(self.device)
            self.actor.load_state_dict(
                torch.load(logs_path_actor, map_location=self.device))
            self.actor.to(self.device)
        elif agentType == 'performance':
            self.critic.load_state_dict(
                torch.load(logs_path_critic, map_location=self.device))
            self.critic.to(self.device)
            self.criticTarget.load_state_dict(
                torch.load(logs_path_critic, map_location=self.device))
            self.criticTarget.to(self.device)
            self.actor.load_state_dict(
                torch.load(logs_path_actor, map_location=self.device))
            self.actor.to(self.device)
        print('  <= Restore {}-{}' .format(logs_path, step))


    def unpack_batch(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
            dtype=torch.bool).to(self.device)
        non_final_state_nxt = torch.FloatTensor([
            s for s in batch.s_ if s is not None]).to(self.device)
        state  = torch.FloatTensor(batch.s).to(self.device)
        action = torch.FloatTensor(batch.a).to(self.device).view(-1, self.actionDim)
        reward = torch.FloatTensor(batch.r).to(self.device)

        g_x = torch.FloatTensor(
            [info['g_x'] for info in batch.info]).to(self.device).view(-1)
        l_x = torch.FloatTensor(
            [info['l_x'] for info in batch.info]).to(self.device).view(-1)

        return non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x
    # * OTHERS ENDS