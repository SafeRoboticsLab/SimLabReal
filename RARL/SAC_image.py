# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import torch
from torch.nn.functional import mse_loss
# from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import os
import time
# import visdom

from .model import SACPiNetwork, SACTwinnedQNetwork
from .ActorCritic import ActorCritic, Transition
import copy

class SAC_image(ActorCritic):
    def __init__(self, CONFIG, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super(SAC_image, self).__init__('SAC', CONFIG)

        #= alpha-related hyper-parameters
        self.init_alpha = CONFIG.ALPHA
        self.LEARN_ALPHA = CONFIG.LEARN_ALPHA
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.target_entropy = -self.actionDim
        self.LR_Al = CONFIG.LR_Al
        self.LR_Al_PERIOD = CONFIG.LR_Al_PERIOD
        self.LR_Al_DECAY = CONFIG.LR_Al_DECAY
        self.LR_Al_END = CONFIG.LR_Al_END
        self.GAMMA_PERIOD = CONFIG.GAMMA_PERIOD
        if self.LEARN_ALPHA:
            print("SAC with learnable alpha and target entropy = {:.1e}".format(
                self.target_entropy))
        else:
            print("SAC with fixed alpha = {:.1e}".format(self.init_alpha))

        #= reach-avoid setting
        self.mode = CONFIG.MODE
        self.terminalType = CONFIG.TERMINAL_TYPE

        #= critic/actor-related hyper-parameters
        self.mlp_dim_actor = CONFIG.MLP_DIM['actor']
        self.mlp_dim_critic = CONFIG.MLP_DIM['critic']
        self.img_sz = CONFIG.IMG_SZ
        self.kernel_sz = CONFIG.KERNEL_SIZE
        self.n_channel = CONFIG.N_CHANNEL
        self.use_ln = CONFIG.USE_LN
        self.use_sm = CONFIG.USE_SM
        self.activation_actor = CONFIG.ACTIVATION['actor']
        self.activation_critic = CONFIG.ACTIVATION['critic']
        self.build_network(verbose=verbose)


    def build_network(self, verbose=True):
        """
        Overriding ActorCritic
        """
        """
        build_network [summary]
        Args:
        """

        # Set up NN
        self.critic = SACTwinnedQNetwork(   input_n_channel=3,
                                            mlp_dim=self.mlp_dim_critic,
                                            actionDim=self.actionDim,
                                            actType=self.activation_critic,
                                            img_sz=self.img_sz,
                                            kernel_sz=self.kernel_sz,
                                            n_channel=self.n_channel,
                                            use_sm=self.use_sm,
                                            use_ln=self.use_ln,
                                            device=self.device,
                                            verbose=verbose
        )
        self.criticTarget = copy.deepcopy(self.critic)
        self.actor = SACPiNetwork(  input_n_channel=3,
                                    mlp_dim=self.mlp_dim_actor,
                                    actionDim=self.actionDim,
                                    actionMag=self.actionMag,
                                    actType=self.activation_actor,
                                    img_sz=self.img_sz,
                                    kernel_sz=self.kernel_sz,
                                    n_channel=self.n_channel,
                                    use_sm=self.use_sm,
                                    use_ln=self.use_ln,
                                    device=self.device,
                                    verbose=verbose
        )
        if verbose:
            print("\nThe actor shares the same encoder with the critic.")
            print('Total parameters in actor: {}'.format(sum(p.numel() for p in self.actor.parameters() if p.requires_grad)))

        # Tie weights for conv layers
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # Set up optimizer
        self.build_optimizer()  # from SAC

        # Initialize alpha
        self.reset_alpha()


    def reset_alpha(self):
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alphaOptimizer = Adam([self.log_alpha], lr=self.LR_Al)
        self.log_alphaScheduler = lr_scheduler.StepLR(self.log_alphaOptimizer,
            step_size=self.LR_Al_PERIOD, gamma=self.LR_Al_DECAY)


    @property
    def alpha(self):
        return self.log_alpha.exp()


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


    def initQ(self, env, warmupIter, outFolder, num_warmup_samples=200,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        loss = 0.0
        lossList = np.empty(warmupIter, dtype=float)
        for ep_tmp in range(warmupIter):
            states, value = env.get_warmup_examples(num_warmup_samples)
            actions = self.genRandomActions(num_warmup_samples)

            self.critic.train()
            value = torch.from_numpy(value).float().to(self.device)
            stateTensor = torch.from_numpy(states).float().to(self.device)
            actionTensor = torch.from_numpy(actions).float().to(self.device)
            q1, q2 = self.critic(stateTensor, actionTensor)
            q1Loss = mse_loss(input=q1, target=value, reduction='sum')
            q2Loss = mse_loss(input=q2, target=value, reduction='sum')
            loss = q1Loss + q2Loss

            self.criticOptimizer.zero_grad()
            loss.backward()
            self.criticOptimizer.step()

            lossList[ep_tmp] = loss.detach().cpu().numpy()
            print('\rWarmup Q [{:d}]. MSE = {:f}'.format(
                ep_tmp+1, loss.detach()), end='')

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
        self.build_optimizer(verbose=False)

        return lossList


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
        # if len(self.memory) < self.start_updates:
        #     return 0.0, 0.0, 0.0, 0.0

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            print('\r{:d}: (q, pi, ent, alpha) = ({:3.5f}/{:3.5f}/{:3.5f}/{:3.5f}).'.format(
                self.cntUpdate, loss_q, loss_pi, loss_entropy, loss_alpha), end=' ')

            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=50,
                MAX_EVAL_EP_STEPS=100,
                warmupBuffer=True, warmupBufferRatio=1.0,
                warmupQ=False, warmupIter=10000,
                optimizeFreq=100, numUpdatePerOptimize=128,
                curUpdates=None, checkPeriod=50000,
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=200,
                storeModel=True, saveBest=False, outFolder='RA',
                useVis=False, verbose=True):

        if useVis:
            import visdom
            vis = visdom.Visdom(env='test_sac_6', port=8098)
            q_loss_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Q Loss'))
            pi_loss_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Pi Loss'))
            entropy_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Entropy'))
            success_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Success'))

        # == Build up networks
        # self.build_network(verbose=verbose)
        # print("Critic is using cuda: ", next(self.critic.parameters()).is_cuda)

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env, ratio=warmupBufferRatio)
        endInitBuffer = time.time()

        # == Warmup Q ==
        startInitQ = time.time()
        if warmupQ:
            self.initQ(env, warmupIter=warmupIter, outFolder=outFolder,
                vmin=vmin, vmax=vmax, plotFigure=plotFigure,
                storeFigure=storeFigure)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainingRecords = []
        trainProgress = []
        checkPointSucc = 0.
        ep = 0

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

            # Rollout
            for _ in range(MAX_EP_STEPS):
                # Select action
                # if warmupBuffer or self.cntUpdate > max(warmupIter, self.start_updates):
                with torch.no_grad():
                    a, _ = self.actor.sample(
                        torch.from_numpy(s).float().to(self.device))
                    a = a.view(-1).cpu().numpy()
                # else:
                    # a = env.action_space.sample()

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
                    def q_func(obs):
                        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                        u = self.actor(obsTensor).detach()
                        v = self.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
                        return v

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

                    if useVis:
                        vis.line(X=array([[self.cntUpdate]]),
                                    Y=array([[success]]),
                                win=success_window,update='append')

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
                            self.save(self.cntUpdate, modelFolder)

                    if plotFigure or storeFigure:
                        # TODO:
                        # fix plot_v error when using critic in RA;
                        # not using it for training performance policy right now
                        if showBool:
                            env.visualize(q_func, policy, vmin=0, boolPlot=True)
                        else:
                            env.visualize(q_func, policy,
                                vmin=vmin, vmax=vmax, cmap='seismic')

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
                        trainingRecords.append([loss_q, loss_pi, loss_entropy, loss_alpha])

                        if timer == 0 and useVis:
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_q]]),
                                    win=q_loss_window,update='append')
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_pi]]),
                                    win=pi_loss_window,update='append')
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_entropy]]),
                                    win=entropy_window,update='append')

                self.cntUpdate += 1

                # Update gamma, lr etc.
                self.updateHyperParam()
                if self.cntUpdate % self.GAMMA_PERIOD == 0 and self.LEARN_ALPHA:
                    self.reset_alpha()

                # Terminate early
                if done:
                    break

        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))

        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress
