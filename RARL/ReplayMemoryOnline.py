# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# import random
import numpy as np
from .ReplayMemory import ReplayMemory

class ReplayMemoryOnline(ReplayMemory):

    def __init__(self, capacity, seed=0):
        super(ReplayMemoryOnline, self).__init__(capacity, seed)
        
        self.memory_online = [] # never full
        self.position_online = 0

    def reset_online(self):
        self.memory_online = []
        self.position_online = 0

    def update_online(self, transition):
        self.memory_online.append(None)
        self.memory_online[self.position_online] = transition
        self.position_online = int((self.position_online + 1))

    def sample_online(self, batch_size):
        length = len(self.memory_online)
        indices = np.random.randint(low=0, high=length, size=(batch_size,))
        return [self.memory_online[i] for i in indices]


    def __len_online__(self):
        return len(self.memory_online)