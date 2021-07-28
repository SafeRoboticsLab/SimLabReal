# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import os
import pickle
import time

from .model import SACPiNetwork, SACTwinnedQNetwork
from .scheduler import StepLRMargin
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model
from .SAC_image import SAC_image
import copy

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class PolicyShieldingJoint(object):
    # region: init
    def __init__(self, CONFIG, CONFIG_PERFORMANCE, CONFIG_BACKUP, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        print("== Constructing performance agent ==")
        self.performance = SAC_image(
            CONFIG_PERFORMANCE['train'], CONFIG_PERFORMANCE['arch'], verbose)
        print("== Constructing backup agent ==")
        self.backup = SAC_image(
            CONFIG_BACKUP['train'], CONFIG_BACKUP['arch'], verbose)
        # probability to activate shielding
        self.eps = StepLRMargin(initValue=CONFIG.EPS,
            period=CONFIG.EPS_PERIOD, decay=CONFIG.EPS_DECAY,
            endValue=CONFIG.EPS_END, goalValue=1.)
        # ratio of training episodes using backup policy
        self.rho = 0.5


    def performanceStateValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.performance.actor(obsTensor).detach()
        v = self.performance.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
        return v


    def backupStateValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.backup.actor(obsTensor).detach()
        v = self.backup.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
        return v


    def learn(  self, env, shieldDict,
                MAX_UPDATES=200000, MAX_EP_STEPS=100, MAX_EVAL_EP_STEPS=100,
                warmupBuffer=True, warmupBufferRatio=1.0,
                optimizeFreq=100, numUpdatePerOptimize=100,
                curUpdates=None, checkPeriod=10000,
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=100,
                storeModel=True, saveBest=False, outFolder='RA', verbose=True):

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env, ratio=warmupBufferRatio)
        endInitBuffer = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainRecords = []
        trainProgress = []
        violationRecord = []
        checkPointSucc = 0.
        ep = 0
        safetyViolationCnt = 0

        if storeModel:
            modelFolder = os.path.join(outFolder, 'model')
            os.makedirs(modelFolder, exist_ok=True)

        if storeFigure:
            figureFolder = os.path.join(outFolder, 'figure')
            os.makedirs(figureFolder, exist_ok=True)

        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))

        if self.mode=='safety':
            endType = 'fail'
        else:
            endType = 'TF'

        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            epCost = np.inf
            ep += 1
            print('\n[{}]: '.format(ep))
            print(env._state)

            # Rollout
            for t in range(MAX_EP_STEPS):
                # Select action
                with torch.no_grad():
                    a, _ = self.actor.sample(
                        torch.from_numpy(s).float().to(self.device))
                    a = a.view(-1).cpu().numpy()

                # Check Safety
                # ? Check if correct
                shieldType = shieldDict['Type']
                if shieldType == 'none':
                    pass
                else:
                    w = env.getTurningRate(a)
                    _state = env.integrate_forward(env._state, w)
                    obs = env._get_obs(_state)

                    if shieldType == 'value':
                        safetyValue = self.backupStateValue(obs)
                        shieldFlag = (safetyValue > shieldDict['Threshold'])
                    elif shieldType == 'simulator':
                        T_ro = shieldDict['T_rollout']
                        _, result, _, _ = env.simulate_one_trajectory(self.actor_backup,
                            T=T_ro, endType='fail', state=_state, latent_prior=None)
                        shieldFlag = (result == -1)

                    if shieldFlag:
                        a = self.actor_backup(s)

                # Interact with env
                s_, r, done, info = env.step(a)
                s_ = None if done else s_
                epCost = max(info["g_x"], min(epCost, info["l_x"]))

                # Store the transition in memory
                self.store_transition(s, a, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    self.actor.eval()
                    self.critic.eval()
                    policy = self.actor  # mean only and no std

                    results = env.simulate_trajectories(policy,
                        T=MAX_EVAL_EP_STEPS, num_rnd_traj=numRndTraj,
                        endType=endType, sample_inside_obs=False,
                        sample_inside_tar=False)[1]
                    if self.mode == 'safety':
                        failure  = np.sum(results==-1)/ results.shape[0]
                        success  =  1 - failure
                        trainProgress.append([success, failure])
                    else:
                        success  = np.sum(results==1) / results.shape[0]
                        failure  = np.sum(results==-1)/ results.shape[0]
                        unfinish = np.sum(results==0) / results.shape[0]
                        trainProgress.append([success, failure, unfinish])

                    if verbose:
                        lr = self.actorOptimizer.state_dict()['param_groups'][0]['lr']
                        print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                        print('  - gamma={:.6f}, lr={:.1e}, alpha={:.1e}.'.format(
                            self.GAMMA, lr, self.alpha))
                        if self.mode == 'safety':
                            print('  - success/failure ratio:', end=' ')
                        else:
                            print('  - success/failure/unfinished ratio:', end=' ')
                        with np.printoptions(formatter={'float': '{: .2f}'.format}):
                            print(np.array(trainProgress[-1]))
                    self.actor.train()
                    self.critic.train()

                    if storeModel:
                        if saveBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, modelFolder)
                        else:
                            self.save(self.cntUpdate, modelFolder, agentType='performance')

                    if plotFigure or storeFigure:
                        if showBool:
                            env.visualize(self.performanceStateValue, policy,
                                vmin=0, boolPlot=True)
                        else:
                            env.visualize(self.performanceStateValue, policy,
                                vmin=vmin, vmax=vmax, cmap='seismic', normalize_v=True)

                        if storeFigure:
                            figurePath = os.path.join(figureFolder,
                                '{:d}.png'.format(self.cntUpdate))
                            plt.savefig(figurePath)
                            plt.close()
                        if plotFigure:
                            plt.show()
                            plt.pause(0.001)
                            plt.close()

                # Perform one step of the optimization (on the target network)
                loss_q, loss_pi, loss_entropy, loss_alpha = 0, 0, 0, 0
                if self.cntUpdate % optimizeFreq == 0:
                    for timer in range(numUpdatePerOptimize):
                        loss_q, loss_pi, loss_entropy, loss_alpha = self.update(timer)
                        trainRecords.append([loss_q, loss_pi, loss_entropy, loss_alpha])

                self.cntUpdate += 1

                # Update gamma, lr etc.
                self.updateHyperParam()
                if self.cntUpdate % self.GAMMA_PERIOD == 0 and self.LEARN_ALPHA:
                    self.reset_alpha()

                # Terminate early
                if done:
                    # g_x = env.safety_margin(env._state, return_boundary=False)
                    # if g_x > 0:
                    safetyViolationCnt += 1
                    violationRecord.append(safetyViolationCnt)
                    break
                if t == (MAX_EP_STEPS-1):
                    violationRecord.append(safetyViolationCnt)

        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, modelFolder, agentType='performance')
        print('\nInitBuffer: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeLearning))

        trainRecords = np.array(trainRecords)
        trainProgress = np.array(trainProgress)
        return trainRecords, trainProgress, violationRecord
    # endregion


    # region: some utils functions
    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, step, logs_path, agentType):
        logs_path_critic = os.path.join(logs_path, agentType, 'critic')
        logs_path_actor = os.path.join(logs_path, agentType, 'actor')
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
        """
        restore

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there should
                be critic/ and agent/ folders.
            agentType (str): performance policy or backup policy.
        """
        logs_path_critic = os.path.join(
            logs_path, agentType, 'critic', 'critic-{}.pth'.format(step))
        logs_path_actor  = os.path.join(
            logs_path, agentType, 'actor',  'actor-{}.pth'.format(step))
        if agentType == 'backup':
            self.critic_backup.load_state_dict(
                torch.load(logs_path_critic, map_location=self.device))
            self.critic_backup.to(self.device)
            if self.train_backup:
                self.criticTarget_backup.load_state_dict(
                    torch.load(logs_path_critic, map_location=self.device))
                self.criticTarget_backup.to(self.device)
            self.actor_backup.load_state_dict(
                torch.load(logs_path_actor, map_location=self.device))
            self.actor_backup.to(self.device)
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
    # endregion