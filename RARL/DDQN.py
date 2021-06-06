# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )


import torch
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.optim as optim

from collections import namedtuple
import os
import pickle

from .model import StepLRMargin, StepResetLR
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])
class DDQN():
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.saved = False
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== PARAM ==
        # Exploration
        self.EpsilonScheduler = StepResetLR(
            initValue=CONFIG.EPSILON, period=CONFIG.EPS_PERIOD, resetPeriod=CONFIG.EPS_RESET_PERIOD,
            decay=CONFIG.EPS_DECAY, endValue=CONFIG.EPS_END)
        self.EPSILON = self.EpsilonScheduler.get_variable()
        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        # Discount Factor
        self.GammaScheduler = StepLRMargin( initValue=CONFIG.GAMMA, period=CONFIG.GAMMA_PERIOD,
                                            decay=CONFIG.GAMMA_DECAY, endValue=CONFIG.GAMMA_END,
                                            goalValue=1.)
        self.GAMMA = self.GammaScheduler.get_variable()
        # Target Network Update
        self.double = CONFIG.DOUBLE
        self.TAU = CONFIG.TAU
        self.HARD_UPDATE = CONFIG.HARD_UPDATE # int, update period
        self.SOFT_UPDATE = CONFIG.SOFT_UPDATE # bool


    def build_network(self):
        raise NotImplementedError


    def build_optimizer(self):
        # self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR_C)
        self.optimizer = torch.optim.AdamW(self.Q_network.parameters(), lr=self.LR_C, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
            step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.max_grad_norm = 1
        self.cntUpdate = 0


    def update(self):
        raise NotImplementedError


    def initBuffer(self, env):
        raise NotImplementedError


    def initQ(self):
        raise NotImplementedError


    def learn(self):
        raise NotImplementedError


    def update_target_network(self):
        if self.SOFT_UPDATE:
            # Soft Replace
            soft_update(self.target_network, self.Q_network, self.TAU)
        elif self.cntUpdate % self.HARD_UPDATE == 0:
            # Hard Replace
            self.target_network.load_state_dict(self.Q_network.state_dict())


    def updateHyperParam(self):
        if self.optimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.scheduler.step()

        self.EpsilonScheduler.step()
        self.EPSILON = self.EpsilonScheduler.get_variable()
        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    def select_action(self):
        raise NotImplementedError


    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path):
        save_model(self.Q_network, step, logs_path, 'Q', self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            self.saved = True


    def restore(self, step, logs_path):
        logs_path = os.path.join(logs_path, 'model', 'Q-{}.pth'.format(step))
        self.Q_network.load_state_dict(
            torch.load(logs_path, map_location=self.device))
        self.target_network.load_state_dict(
            torch.load(logs_path, map_location=self.device))
        print('  => Restore {}' .format(logs_path))
