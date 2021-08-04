# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

# We train a performance policy safely given a backup policy trained in similar
# environment(s). Here we consider two shielding criteria: (1) using a forward
# simulator to check if from the new state we can remain safe within T_ro steps
# and (2) using safety critic values. We check in every T_ch steps. Here the
# backup policy is a priori. The immediate next step is training the backup and
# performance policy together.

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
import time

from .model import SACPiNetwork, SACTwinnedQNetwork
from .scheduler import StepLRMargin
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model
import copy

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'info'])


class PolicyShielding(object):
    # region: init
    def __init__(self, CONFIG, CONFIG_PERFORMANCE, CONFIG_BACKUP, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        self.mode = 'performance'
        self.CONFIG = CONFIG
        self.CONFIG_PERFORMANCE = CONFIG_PERFORMANCE
        self.CONFIG_BACKUP = CONFIG_BACKUP
        self.saved = False
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

        #== ENV PARAM ==
        self.obsChannel = CONFIG.OBS_CHANNEL
        self.actionMag = CONFIG.ACTION_MAG
        self.actionDim = CONFIG.ACTION_DIM
        self.img_sz = CONFIG.IMG_SZ

        #== PARAM ==
        #= Learning
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE
        self.TAU = CONFIG.TAU  #= Target Network Update
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
        self.critic = SACTwinnedQNetwork(
            input_n_channel=self.obsChannel,
            img_sz=self.img_sz,
            actionDim=self.actionDim,
            mlp_dim=CONFIG.MLP_DIM['critic'],
            actType=CONFIG.ACTIVATION['critic'],
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
        self.actor = SACPiNetwork(
            input_n_channel=self.obsChannel,
            img_sz=self.img_sz,
            actionDim=self.actionDim,
            actionMag=self.actionMag,
            mlp_dim=CONFIG.MLP_DIM['actor'],
            actType=CONFIG.ACTIVATION['actor'],
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
        self.critic_backup = SACTwinnedQNetwork(
            input_n_channel=self.obsChannel,
            img_sz=self.img_sz,
            actionDim=self.actionDim,
            mlp_dim=CONFIG.MLP_DIM['critic'],
            actType=CONFIG.ACTIVATION['critic'],
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
        self.actor_backup = SACPiNetwork(
            input_n_channel=self.obsChannel,
            img_sz=self.img_sz,
            actionDim=self.actionDim,
            actionMag=self.actionMag,
            mlp_dim=CONFIG.MLP_DIM['actor'],
            actType=CONFIG.ACTIVATION['actor'],
            kernel_sz=CONFIG.KERNEL_SIZE,
            n_channel=CONFIG.N_CHANNEL,
            use_sm=CONFIG.USE_SM,
            use_ln=CONFIG.USE_LN,
            device=self.device,
            verbose=verbose
        )

        # Tie weights for conv layers
        self.actor_backup.encoder.copy_conv_weights_from(self.critic_backup.encoder)


    def build_optimizer(self):
        self.criticOptimizer = Adam(self.critic.parameters(), lr=self.LR_C)
        self.actorOptimizer = Adam(self.actor.parameters(), lr=self.LR_A)

        self.criticScheduler = lr_scheduler.StepLR(self.criticOptimizer,
            step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY)
        self.actorScheduler = lr_scheduler.StepLR(self.actorOptimizer,
            step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY)

        if self.LEARN_ALPHA:
            self.log_alpha.requires_grad = True
            self.log_alphaOptimizer = Adam([self.log_alpha], lr=self.LR_Al)
            self.log_alphaScheduler = lr_scheduler.StepLR(
                self.log_alphaOptimizer,
                step_size=self.LR_Al_PERIOD, gamma=self.LR_Al_DECAY)

        self.max_grad_norm = .1
        self.cntUpdate = 0


    def reset_alpha(self):
        # print("Reset alpha")
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alphaOptimizer = Adam([self.log_alpha], lr=self.LR_Al)
        self.log_alphaScheduler = lr_scheduler.StepLR(self.log_alphaOptimizer,
            step_size=self.LR_Al_PERIOD, gamma=self.LR_Al_DECAY)


    @property
    def alpha(self):
        return self.log_alpha.exp()
    # endregion


    # region: learning-related
    def initBuffer(self, env, ratio=1.):
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


    def update_critic_hyperParam(self):
        if self.criticOptimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_C_END:
            for param_group in self.criticOptimizer.param_groups:
                param_group['lr'] = self.LR_C_END
        else:
            self.criticScheduler.step()

        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    def update_actor_hyperParam(self):
        if self.actorOptimizer.state_dict()['param_groups'][0]['lr'] <= self.LR_A_END:
            for param_group in self.actorOptimizer.param_groups:
                param_group['lr'] = self.LR_A_END
        else:
            self.actorScheduler.step()


    def update_alpha_hyperParam(self):
        lr = self.log_alphaOptimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.LR_Al_END:
            for param_group in self.log_alphaOptimizer.param_groups:
                param_group['lr'] = self.LR_Al_END
        else:
            self.log_alphaScheduler.step()


    def updateHyperParam(self):
        self.update_critic_hyperParam()
        self.update_actor_hyperParam()
        if self.LEARN_ALPHA:
            self.update_alpha_hyperParam()


    def update_target_networks(self):
        soft_update(self.criticTarget, self.critic, self.TAU)


    def update_critic(self, batch):
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x = \
            self.unpack_batch(batch)
        self.critic.train()
        self.criticTarget.eval()
        self.actor.eval()

        #== get Q(s,a) ==
        q1, q2 = self.critic(state, action)  # Used to compute loss (non-target part).

        #== placeholder for target ==
        y = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        #== compute actor next_actions and feed to criticTarget ==
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(non_final_state_nxt)
            next_q1, next_q2 = self.criticTarget(non_final_state_nxt, next_actions)

            if self.mode == 'RA' or self.mode=='safety': # use max for RA or safety
                q_max = torch.max(next_q1, next_q2).view(-1)
            elif self.mode == 'performance':
                q_min = torch.min(next_q1, next_q2).view(-1)
            else:
                raise ValueError("Unsupported RL mode.")

            if self.mode == 'RA':
                y[non_final_mask] =  (
                    (1.0 - self.GAMMA) * torch.max(l_x[non_final_mask], g_x[non_final_mask]) +
                    self.GAMMA * torch.max( g_x[non_final_mask], torch.min(l_x[non_final_mask], q_max)))
                if self.terminalType == 'g':
                    y[torch.logical_not(non_final_mask)] = g_x[torch.logical_not(non_final_mask)]
                elif self.terminalType == 'max':
                    y[torch.logical_not(non_final_mask)] = torch.max(
                        l_x[torch.logical_not(non_final_mask)], g_x[torch.logical_not(non_final_mask)])
                else:
                    raise ValueError("invalid terminalType")
            elif self.mode == 'safety':
                # V(s) = max{ g(s), V(s') }, Q(s, u) = V( f(s,u) )
                # normal state
                y[non_final_mask] =  (
                    (1.0 - self.GAMMA) * g_x[non_final_mask] +
                    self.GAMMA * torch.max( g_x[non_final_mask], q_max))

                # terminal state
                final_mask = torch.logical_not(non_final_mask)
                y[final_mask] = g_x[final_mask]
            elif self.mode == 'performance':
                target_q = q_min - self.alpha * next_log_prob.view(-1)  # already masked - can be lower dim than y
                y = reward
                y[non_final_mask] += self.GAMMA*target_q

        #== MSE update for both Q1 and Q2 ==
        loss_q1 = mse_loss(input=q1.view(-1), target=y)
        loss_q2 = mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        #== backpropagation ==
        self.criticOptimizer.zero_grad()
        loss_q.backward()
        self.criticOptimizer.step()

        return loss_q.item()


    def update_actor(self, batch):
        """
        Use detach_encoder=True to not update conv layers
        """
        _, _, state, _, _, _, _ = self.unpack_batch(batch)

        self.critic.eval()
        self.actor.train()

        action_sample, log_prob = self.actor.sample(state, detach_encoder=True)
        q_pi_1, q_pi_2 = self.critic(state, action_sample, detach_encoder=True)

        if self.mode == 'RA' or self.mode=='safety':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)

        # Obj: min_theta E[ Q(s, pi_theta(s)) + alpha * log(pi_theta(s))]
        # loss_pi = (q_pi + self.alpha * log_prob.view(-1)).mean()
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode=='safety':
            loss_pi = loss_q_eval + self.alpha * loss_entropy
        elif self.mode == 'performance':
            loss_pi = -loss_q_eval + self.alpha * loss_entropy
        self.actorOptimizer.zero_grad()
        loss_pi.backward()
        self.actorOptimizer.step()

        # Automatic temperature tuning
        loss_alpha = (self.alpha *
            (-log_prob - self.target_entropy).detach()).mean()
        if self.LEARN_ALPHA:
            self.log_alphaOptimizer.zero_grad()
            loss_alpha.backward()
            self.log_alphaOptimizer.step()
        return loss_pi.item(), loss_entropy.item(), loss_alpha.item()


    def update(self, timer, update_period=2):
        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            # print('\r{:d}: (q, pi, ent, alpha) = ({:3.5f}/{:3.5f}/{:3.5f}/{:3.5f}).'.format(
            #     self.cntUpdate, loss_q, loss_pi, loss_entropy, loss_alpha), end=' ')
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha


    def performanceStateValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.actor(obsTensor).detach()
        v = self.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
        return v


    def backupStateValue(self, obs):
        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        u = self.actor_backup(obsTensor).detach()
        v = self.critic_backup(obsTensor, u)[0].cpu().detach().numpy()[0]
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
        cntSafetyViolation = 0

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
                    g_x = env.safety_margin(env._state, return_boundary=False)
                    if g_x > 0:
                        cntSafetyViolation += 1
                    violationRecord.append(cntSafetyViolation)
                    break
                if t == (MAX_EP_STEPS-1):
                    violationRecord.append(cntSafetyViolation)

            print('  - This episode has {} steps'.format(t))
            print('  - Safety violations so far: {:d}'.format(cntSafetyViolation))

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
            logs_path, 'critic', 'critic-{}.pth'.format(step))
        logs_path_actor  = os.path.join(
            logs_path, 'actor',  'actor-{}.pth'.format(step))
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
    # endregion