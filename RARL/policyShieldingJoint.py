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

from .scheduler import StepLRMargin, StepLR
from .ReplayMemory import ReplayMemory
# from .utils import save_model
from .SAC_mini import SAC_mini

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class PolicyShieldingJoint(object):
    def __init__(self, CONFIG, CONFIG_PERFORMANCE, CONFIG_BACKUP, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
        self.saved = False
        self.device = CONFIG.DEVICE
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.CONFIG = CONFIG
        # self.CONFIG_PERFORMANCE = CONFIG_PERFORMANCE
        # self.CONFIG_BACKUP = CONFIG_BACKUP

        print("= Constructing performance agent")
        self.performance = SAC_mini(
            CONFIG_PERFORMANCE['train'], CONFIG_PERFORMANCE['arch'], verbose)

        print("= Constructing backup agent")
        self.backup = SAC_mini(
            CONFIG_BACKUP['train'], CONFIG_BACKUP['arch'], verbose)

        # probability to activate shielding: -> 1 and in the timescale of episodes
        self.EpsilonScheduler = StepLRMargin(initValue=CONFIG.EPS,
            period=CONFIG.EPS_PERIOD, decay=CONFIG.EPS_DECAY,
            endValue=CONFIG.EPS_END, goalValue=1.)
        self.EPS = self.EpsilonScheduler.get_variable()

        # ratio of episodes using backup policy: -> 0.1 and in the timescale of episodes
        self.RhoScheduler = StepLR(initValue=CONFIG.RHO,
            period=CONFIG.RHO_PERIOD, decay=CONFIG.RHO_DECAY,
            endValue=CONFIG.RHO_END)
        self.RHO = self.RhoScheduler.get_variable()


    def initBuffer(self, env, ratio=1.0):
        cnt = 0
        s = env.reset()
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            a = env.action_space.sample()
            # a = self.genRandomActions(1)[0]
            s_, r, done, info = env.step(a)
            s_ = None if done else s_
            self.store_transition(s, a, r, s_, info)
            if done:
                s = env.reset()
            else:
                s = s_
        print(" --- Warmup Buffer Ends")


    def learn(  self, env, shieldDict,
                MAX_STEPS=200000, MAX_EP_STEPS=100, MAX_EVAL_EP_STEPS=100,
                warmupBuffer=True, warmupBufferRatio=1.0,
                optimizeFreq=100, numUpdatePerOptimize=100,
                curSteps=None, checkPeriod=10000,
                plotFigure=True, storeFigure=False,
                vmin=-1, vmax=1, numRndTraj=100,
                storeModel=True, saveBest=False, outFolder='RA', verbose=True):

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env, ratio=warmupBufferRatio)
        endInitBuffer = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainRecords = [[], []]
        trainProgress = [[], []]
        violationRecord = []
        cpSuccBackup = 0.
        cpSuccPerf = 0.
        ep = 0
        cntSafetyViolation = 0

        if storeModel:
            modelFolder = os.path.join(outFolder, 'model')
            modelFolderPerf   = os.path.join(modelFolder, 'performance')
            modelFolderBackup = os.path.join(modelFolder, 'backup')
            os.makedirs(modelFolderPerf, exist_ok=True)
            os.makedirs(modelFolderBackup, exist_ok=True)
            self.save(modelFolder)

        if storeFigure:
            figFolderPerf   = os.path.join(outFolder, 'figure', 'performance')
            figFolderBackup = os.path.join(outFolder, 'figure', 'backup')
            os.makedirs(figFolderPerf, exist_ok=True)
            os.makedirs(figFolderBackup, exist_ok=True)

        if curSteps is None:
            self.cntStep = 0
        else:
            self.cntStep = curSteps
            print("starting from {:d} steps".format(self.cntStep))

        shieldType = shieldDict['Type']
        # print(shieldType)
        assert (shieldType == 'value') or (shieldType == 'simulator'),\
            'Invalid Shielding Type!'

        while self.cntStep <= MAX_STEPS:
            s = env.reset()
            ep += 1
            print('\nAfter [{:d}] episodes:'.format(ep))
            print(env._state)
            # Choose which policy for this training episode
            if (np.random.rand() < self.RHO):
                on_policy = self.backup
                use_perf = False
                print("  - Use backup policy")
            else:
                on_policy = self.performance
                use_perf = True
                print("  - Use performance policy")
            apply_shielding = (np.random.rand() < self.EPS) and (use_perf)
            if apply_shielding:
                print("  - Shielding activated in this episode")

            # Rollout
            for t in range(MAX_EP_STEPS):
                # Select action
                with torch.no_grad():
                    a, _ = on_policy.actor.sample(
                        torch.from_numpy(s).float().to(self.device))
                    a = a.view(-1).cpu().numpy()

                # Shielding
                if apply_shielding:
                    # get the next state
                    w = env.getTurningRate(a)
                    _state = env.integrate_forward(env._state, w)
                    obs = env._get_obs(_state)
                    if shieldType == 'value':
                        safetyValue = self.backupValue(obs)
                        shieldFlag = (safetyValue > shieldDict['Threshold'])
                    elif shieldType == 'simulator':
                        T_ro = shieldDict['T_rollout']
                        _, result, _, _ = env.simulate_one_trajectory(self.backup.actor,
                            T=T_ro, endType='fail', state=_state, latent_prior=None)
                        shieldFlag = (result == -1)
                    if shieldFlag:
                        a = self.backup.actor(s)

                # Interact with env
                s_, r, done, info = env.step(a)
                s_ = None if done else s_

                # Store the transition in memory
                self.store_transition(s, a, r, s_, info)
                s = s_

                # Check after fixed number of steps
                if self.cntStep != 0 and self.cntStep % checkPeriod == 0:
                    progressPerf = self.performance.check(env, self.cntStep,
                        MAX_EVAL_EP_STEPS, numRndTraj, verbose=True)
                    progressBackup = self.backup.check(env, self.cntStep,
                        MAX_EVAL_EP_STEPS, numRndTraj, verbose=True)
                    trainProgress[0].append(progressPerf)
                    trainProgress[1].append(progressBackup)

                    if storeModel:
                        if saveBest:
                            if progressPerf[0] > cpSuccPerf:
                                cpSuccPerf = progressPerf[0]
                                self.performance.save(self.cntStep, modelFolderPerf)
                            if progressBackup[0] > cpSuccBackup:
                                cpSuccBackup = progressBackup[0]
                                self.backup.save(self.cntStep, modelFolderBackup)
                        else:
                            self.performance.save(self.cntStep, modelFolderPerf)
                            self.backup.save(self.cntStep, modelFolderBackup)

                    if plotFigure or storeFigure:
                        figPerf = env.visualize(self.performanceValue,
                            self.performance.actor, vmin=vmin, vmax=vmax,
                            cmap='seismic', normalize_v=True)
                        figBackup = env.visualize(self.backupValue,
                            self.backup.actor, vmin=vmin, vmax=vmax, cmap='seismic')

                        if storeFigure:
                            figPerf.savefig(os.path.join(
                                figFolderPerf,   '{:d}.png'.format(self.cntStep))
                            )
                            figBackup.savefig(os.path.join(
                                figFolderBackup, '{:d}.png'.format(self.cntStep))
                            )
                            plt.close()
                        if plotFigure:
                            figPerf.show()
                            figBackup.show()
                            plt.pause(0.01)
                            plt.close()

                # Time to update
                loss_q, loss_pi, loss_entropy, loss_alpha = 0, 0, 0, 0
                if self.cntStep % optimizeFreq == 0:
                    for timer in range(numUpdatePerOptimize):
                        transitions = self.memory.sample(self.BATCH_SIZE)
                        batch = Transition(*zip(*transitions))
                        loss_q, loss_pi, loss_entropy, loss_alpha = \
                            self.performance.update(batch, timer, update_period=2)
                        trainRecords[0].append([loss_q, loss_pi, loss_entropy, loss_alpha])
                        loss_q, loss_pi, loss_entropy, loss_alpha = \
                            self.backup.update(batch, timer, update_period=2)
                        trainRecords[1].append([loss_q, loss_pi, loss_entropy, loss_alpha])

                self.cntStep += 1

                # Update gamma, lr etc.
                self.performance.updateHyperParam()
                self.backup.updateHyperParam()

                # Terminate early
                if done:
                    g_x = env.safety_margin(env._state, return_boundary=False)
                    if g_x > 0:
                        cntSafetyViolation += 1
                    violationRecord.append(cntSafetyViolation)
                    break
                if t == (MAX_EP_STEPS-1):
                    violationRecord.append(cntSafetyViolation)

            # Update epsilon, rho
            print('  - This episode has {} steps'.format(t))
            print('  - Safety violations so far: {:d}'.format(cntSafetyViolation))
            print('  - eps={:.2f}, rho={:.2f}'.format(self.EPS, self.RHO))
            self.EpsilonScheduler.step()
            self.EPS = self.EpsilonScheduler.get_variable()
            self.RhoScheduler.step()
            self.RHO = self.RhoScheduler.get_variable()

        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeLearning = endLearning - startLearning
        print('\nInitBuffer: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeLearning))

        trainRecords = np.array(trainRecords)
        trainProgress[0] = np.stack( trainProgress[0], axis=0 )
        trainProgress[1] = np.stack( trainProgress[1], axis=0 )
        return trainRecords, trainProgress, violationRecord


    # region: some utils functions
    def performanceValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.performance.actor(obsTensor).detach()
        v = self.performance.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
        return v


    def backupValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.backup.actor(obsTensor).detach()
        v = self.backup.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
        return v


    def store_transition(self, *args):
        self.memory.update(Transition(*args))


    def save(self, logs_path):
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            self.saved = True


    # def save(self, step, logs_path, agentType):
    #     path_c = os.path.join(logs_path, agentType, 'critic')
    #     path_a = os.path.join(logs_path, agentType, 'actor')
    #     if agentType == 'backup':
    #         save_model(self.backup.critic, step, path_c, 'critic', self.MAX_MODEL)
    #         save_model(self.backup.actor,  step, path_a, 'actor',  self.MAX_MODEL)
    #     elif agentType == 'performance':
    #         save_model(self.performance.critic, step, path_c, 'critic', self.MAX_MODEL)
    #         save_model(self.performance.actor,  step, path_a, 'actor',  self.MAX_MODEL)
    #     if not self.saved:
    #         config_path = os.path.join(logs_path, "CONFIG.pkl")
    #         pickle.dump(self.CONFIG, open(config_path, "wb"))
    #         config_path = os.path.join(logs_path, "CONFIG_PERFORMANCE.pkl")
    #         pickle.dump(self.CONFIG_PERFORMANCE, open(config_path, "wb"))
    #         config_path = os.path.join(logs_path, "CONFIG_BACKUP.pkl")
    #         pickle.dump(self.CONFIG_BACKUP, open(config_path, "wb"))
    #         self.saved = True


    def restore(self, step, logs_path, agentType):
        """
        restore

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
            agentType (str): performance policy or backup policy.
        """
        modelFolder = path_c = os.path.join(logs_path, agentType)
        path_c = os.path.join(
            modelFolder, 'critic', 'critic-{}.pth'.format(step))
        path_a  = os.path.join(
            modelFolder, 'actor',  'actor-{}.pth'.format(step))
        if agentType == 'backup':
            self.backup.critic.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.backup.critic.to(self.device)
            self.backup.criticTarget.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.backup.criticTarget.to(self.device)
            self.backup.actor.load_state_dict(
                torch.load(path_a, map_location=self.device))
            self.backup.actor.to(self.device)
        elif agentType == 'performance':
            self.performance.critic.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.performance.critic.to(self.device)
            self.performance.criticTarget.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.performance.criticTarget.to(self.device)
            self.performance.actor.load_state_dict(
                torch.load(path_a, map_location=self.device))
            self.performance.actor.to(self.device)
        print('  <= Restore {} with {} updates from {}.'.format(
            agentType, step, modelFolder))
    # endregion