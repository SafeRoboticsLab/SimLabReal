# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for replay buffers.
"""

import numpy as np
from collections import deque


class ReplayMemory(object):

    def __init__(self, capacity, seed):
        self.reset(capacity)
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def reset(self, capacity):
        if capacity is None:
            capacity = self.capacity
        self.memory = deque(maxlen=capacity)

    def update(self, transition):
        self.memory.appendleft(transition)  # pop from right if full

    def sample(
        self,
        batch_size,
        traj_size=None,  # assume traj saved are same length
        frame_skip=None,
        recent_size=None,
    ):
        length = len(self.memory)
        if recent_size is not None:
            length = min(length, recent_size)
        indices = self.rng.integers(low=0, high=length, size=(batch_size,))
        return [self.memory[i] for i in indices], None  # dummy for nxt

    def __len__(self):
        return len(self.memory)


class ReplayMemoryTraj():

    def __init__(self, capacity, seed, sample_next=False):
        self.reset(capacity)
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.sample_next = sample_next

    def reset(self, capacity=None):
        if capacity is None:
            capacity = self.capacity
        self.memory = deque(maxlen=capacity)
        self.traj_len = deque(maxlen=capacity)

    def update(self, traj):
        self.memory.appendleft(traj)  # pop from right if full
        self.traj_len.appendleft(len(traj))

    def set_possible_samples(
        self, traj_size=50, frame_skip=0, allow_repeat_frame=False,
        recent_size=None
    ):
        # if burn-in, not using initial steps, might be an issue with
        # fixed_init also some trajectories can be too short for traj_size
        if allow_repeat_frame:
            self.offset = 0
        else:
            self.offset = (traj_size-1) * frame_skip + traj_size
        if recent_size is not None:
            traj_len_all = [
                self.traj_len[ind]
                for ind in range(min(recent_size, len(self.traj_len)))
            ]
        else:
            traj_len_all = self.traj_len
        self.possible_end_inds = []
        for traj_ind, traj_len in enumerate(
            traj_len_all
        ):  # this is fine since recent traj starts from ind=0 (from the left)
            self.possible_end_inds += [
                (traj_ind, transition_ind)
                for transition_ind in range(self.offset, traj_len)
            ]  # allow done at the end

    def sample(self, batch_size, traj_size=50, frame_skip=0):
        # min steps needed; if fewer, randomly sample
        traj_cover = (traj_size-1) * frame_skip + traj_size
        inds = self.rng.integers(
            low=0, high=len(self.possible_end_inds), size=(batch_size,)
        )
        out = []
        out_nxt = []
        for ind in inds:
            traj_ind, transition_ind = self.possible_end_inds[ind]

            # Implicitly allow repeat frame
            if transition_ind < traj_cover:
                if transition_ind == 0:
                    seq = np.zeros((traj_size), dtype='int')
                else:  # randomly sampled from
                    seq_random = np.random.choice(
                        transition_ind, traj_size - 1, replace=True
                    )  # exclude transition_ind
                    seq_random = np.sort(seq_random)  # ascending
                    seq = np.append(seq_random, transition_ind)  # add to end
            else:
                seq = -np.arange(0,
                                 traj_size) * (frame_skip+1) + transition_ind
                seq = np.flip(seq, 0)
            out += [[self.memory[traj_ind][ind] for ind in seq]]

            # Get next - can be empty if prev is done
            if self.sample_next:
                transition_ind += 1
                if transition_ind < traj_cover:  # cannot be 0 any more
                    seq_random = np.random.choice(
                        transition_ind, traj_size - 1, replace=True
                    )
                    seq_random = np.sort(seq_random)
                    seq = np.append(seq_random, transition_ind)
                elif transition_ind == self.traj_len[traj_ind]:
                    seq = []
                else:
                    seq = -np.arange(0, traj_size
                                    ) * (frame_skip+1) + transition_ind
                    seq = np.flip(seq, 0)
                out_nxt += [[self.memory[traj_ind][ind] for ind in seq]]
        return out, out_nxt

    def __len__(self):
        return len(self.memory)

    @property
    def num_sample(self):
        return sum(self.traj_len)
