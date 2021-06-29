# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.utils.data as Data

import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from .model import DeterministicPolicy, TwinnedQNetwork
from .ActorCritic import ActorCritic

class TD3_image(ActorCritic):
    def __init__(self, CONFIG, actionSpace, dimLists, terminalType='g', verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            actionSpace (Class object): consists of `high` and `low` attributes.
            dimList (list): consists of dimension lists
            actType (dict, optional): consists of activation types.
                Defaults to ['Tanh', 'Tanh'].
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super(TD3_image, self).__init__('TD3', CONFIG, actionSpace)
        self.terminalType = terminalType

        #== Build NN for (D)DQN ==
        assert dimLists is not None, "Define the architectures"
        self.dimListCritic = dimLists[0]
        self.dimListActor = dimLists[1]
        self.actType = CONFIG.ACTIVATION
        self.build_network(dimLists, self.actType, verbose=verbose)


    def build_critic(self, dimList, actType='Tanh', verbose=True):
        self.critic = TwinnedQNetwork(
                        dimList=dimList, 
                        actType=actType, 
                        device=self.device,
                        verbose=verbose, 
                        image=True,
                        actionDim=self.actionDim
        )
        self.criticTarget = deepcopy(self.critic)
        for p in self.criticTarget.parameters():
            p.requires_grad = False


    def build_actor(self, dimListActor, actType='Tanh', noiseStd=0.2,
        noiseClamp=0.5, verbose=True):
        self.actor = DeterministicPolicy(dimListActor, self.actionSpace,
            actType=actType, noiseStd=noiseStd, noiseClamp=noiseClamp,
            device=self.device, image=True, verbose=verbose)
        self.actorTarget = deepcopy(self.actor)
        for p in self.actorTarget.parameters():
            p.requires_grad = False


    def initBuffer(self, env, ratio=1.):
        cnt = 0
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            s = env.reset()
            a = self.actionSpace.sample()
            s_, r, done, info = env.step(a)
            s_ = None if done else s_
            self.store_transition(s, a, r, s_, info)
        print(" --- Warmup Buffer Ends")


    def initQ(self, env, warmupIter, outFolder, num_warmup_samples=5000,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        loss = 0.0
        # lossList = np.empty(warmupIter, dtype=float)
        lossArray = []
        states, value = env.get_warmup_examples(num_warmup_samples)
        actions = self.genRandomActions(num_warmup_samples)
        states = torch.FloatTensor(states).to(self.device)
        value = torch.FloatTensor(value).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        heuristic_dataset = Data.TensorDataset(states, actions, value)
        heuristic_dataloader = Data.DataLoader( heuristic_dataset,
                                                batch_size=32,
                                                shuffle=True)

        self.critic.train()
        for ep_tmp in range(warmupIter):
            i = 0
            lossList = []
            for stateTensor, actionTensor, valueTensor in heuristic_dataloader:
                i += 1
                q1, q2 = self.critic(stateTensor, actionTensor)
                q1Loss = mse_loss(input=q1, target=valueTensor, reduction='sum')
                q2Loss = mse_loss(input=q2, target=valueTensor, reduction='sum')
                loss = q1Loss + q2Loss

                self.criticOptimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.criticOptimizer.step()

                # lossList[ep_tmp] = loss.detach().cpu().numpy()
                lossList.append(loss.detach().cpu().numpy())
                print('\rWarmup Q [{:d}-{:d}]. MSE = {:f}'.format(
                    ep_tmp+1, i, loss.detach()), end='')
            lossArray.append(lossList)

        print(" --- Warmup Q Ends")
        if plotFigure or storeFigure:
            env.visualize(self.critic.Q1, self.actor, vmin=vmin, vmax=vmax)
            if storeFigure:
                figureFolder = os.path.join(outFolder, 'figure')
                os.makedirs(figureFolder, exist_ok=True)
                figurePath = os.path.join(figureFolder, 'initQ.png')
                plt.savefig(figurePath)
            if plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()

        # hard replace
        self.criticTarget.load_state_dict(self.critic.state_dict())
        del self.criticOptimizer
        self.build_optimizer()

        return np.array(lossArray)


    def update_critic(self, batch):

        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x = \
            self.unpack_batch(batch)
        final_mask = torch.logical_not(non_final_mask)

        self.critic.train()
        self.criticTarget.eval()
        self.actorTarget.eval()

        #== get Q(s,a) ==
        q1, q2 = self.critic(state, action)  # Used to compute loss (non-target part).

        #== placeholder for target ==
        target_q = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        #== compute actorTarget next_actions and feed to criticTarget ==
        with torch.no_grad():
            # clip(pi_targ(s')+clip(eps,-c,c),a_low, a_high)
            _, next_actions = self.actorTarget.sample(non_final_state_nxt)
            next_q1, next_q2 = self.criticTarget(non_final_state_nxt, next_actions)
            # max because we are doing reach-avoid
            q_max = torch.max(next_q1, next_q2).view(-1)

        target_q[non_final_mask] =  (
            (1.0 - self.GAMMA) * torch.max(l_x[non_final_mask], g_x[non_final_mask]) +
            self.GAMMA * torch.max( g_x[non_final_mask],
                                    torch.min(l_x[non_final_mask], q_max))
        )

        if self.terminalType == 'g':
            target_q[final_mask] = g_x[final_mask]
        elif self.terminalType == 'max':
            target_q[final_mask] = torch.max(l_x[final_mask], g_x[final_mask])
        else:
            raise ValueError("invalid terminalType")

        #== MSE update for both Q1 and Q2 ==
        loss_q1 = mse_loss(input=q1.view(-1), target=target_q)
        loss_q2 = mse_loss(input=q2.view(-1), target=target_q)
        loss_q = loss_q1 + loss_q2

        #== backpropagation ==
        self.criticOptimizer.zero_grad()
        loss_q.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.criticOptimizer.step()

        return loss_q.item()


    def update_actor(self, batch):

        state = self.unpack_batch(batch)[2]

        self.critic.eval()
        self.actor.train()
        for p in self.critic.parameters():
            p.requires_grad = False

        q_pi_1, q_pi_2 = self.critic(state, self.actor(state))
        q_pi = torch.max(q_pi_1, q_pi_2)

        loss_pi = q_pi.mean()
        self.actorOptimizer.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actorOptimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return loss_pi.item()