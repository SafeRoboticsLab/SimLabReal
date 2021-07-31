# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import torch
from torch.nn.functional import mse_loss
# from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import os
import time
from collections import namedtuple
import copy
import math

from .model import Discriminator, SACPiNetwork, SACTwinnedQNetwork
from .SAC_image import SAC_image
from .ReplayMemoryOnline import ReplayMemoryOnline

# Include latent variable in buffer
TransitionLatent = namedtuple('TransitionLatent', ['z', 's', 'a', 'r', 's_', 'info'])


class SAC_image_maxEnt(SAC_image):
    def __init__(self, CONFIG, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super(SAC_image_maxEnt, self).__init__(CONFIG)
        self.share_disc_encoder_weight = False
        self.actor_freq = 1

        # Overwrite replay buffer initialization
        self.memory = ReplayMemoryOnline(CONFIG.MEMORY_CAPACITY)    

        # Discriminator training
        self.mlp_dim_disc = [256, 256]
        self.LR_D = CONFIG.LR_D
        self.fit_freq = CONFIG.FIT_FREQ
        self.fit_freq_rate = 1.0   # every optimize
        self.aug_reward_range = CONFIG.AUG_REWARD_RANGE
        self.disc_batch_size = 64

        # Episode initialization
        self.fixed_init = CONFIG.FIXED_INIT
    
        # Latent
        self.log_ps_bound = 20.0    #! this depends on latent dimension, but too big should be fine
        self.latent_prior_std = CONFIG.LATENT_PRIOR_STD
        self.latent_dim = CONFIG.LATENT_DIM
        self.latent_mean = torch.zeros((self.latent_dim)).to(self.device)
        self.latent_std = self.latent_prior_std*torch.ones((self.latent_dim)).to(self.device) # unit gaussian
        self.latent_prior = torch.distributions.Normal(self.latent_mean, self.latent_std)


    def sample_from_prior(self, size=torch.Size(), return_log_prob=False):
        latent = self.latent_prior.sample(size) # no gradient unlike rsample()
        if return_log_prob:
            log_prob = self.latent_prior.log_prob(latent)
            return latent, log_prob
        return latent


    def build_network(self, verbose=True):
        """
        Overriding SAC_image
        """
        self.critic = SACTwinnedQNetwork(   input_n_channel=3,
                                            latent_dim=self.latent_dim,
                                            mlp_dim=self.mlp_dim_critic,
                                            actionDim=self.actionDim,
                                            actType=self.activation_critic,
                                            img_sz=self.img_sz,
                                            kernel_sz=self.kernel_sz,
                                            n_channel=self.n_channel,
                                            device=self.device,
                                            verbose=verbose
        )
        self.criticTarget = copy.deepcopy(self.critic)
        self.actor = SACPiNetwork(  input_n_channel=3,
                                    latent_dim=self.latent_dim,
                                    mlp_dim=self.mlp_dim_actor,
                                    actionDim=self.actionDim,
                                    actionMag=self.actionMag,
                                    actType=self.activation_actor,
                                    img_sz=self.img_sz,
                                    kernel_sz=self.kernel_sz,
                                    n_channel=self.n_channel,
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


    def setup_disc(self, verbose=True):
        self.disc = Discriminator(input_n_channel=3*2,    # stack frames
                                    latent_dim=self.latent_dim,
                                    mlp_dim=self.mlp_dim_actor,
                                    img_sz=self.img_sz,
                                    kernel_sz=self.kernel_sz,
                                    n_channel=self.n_channel,
                                    device=self.device,
                                    verbose=verbose)

        # Tie weights for conv layers
        if self.share_disc_encoder_weight:
            self.disc.encoder.copy_conv_weights_from(self.critic.encoder)

        self.discOptimizer = Adam(self.disc.parameters(), lr=self.LR_D)   # no schedule now


    def store_transition(self, *args):
        self.memory.update(TransitionLatent(*args))


    def store_transition_online(self, *args):
        self.memory.update_online(TransitionLatent(*args))


    def initBuffer(self, env, ratio=1.0):
        cnt = 0
        s = env.reset(random_init=not self.fixed_init)
        z = self.sample_from_prior().view(1,-1).to(self.device) # doesn't matter
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            a = env.action_space.sample()
            s_, r, done, info = env.step(a)
            info['init'] = True #! indicate not using z
            s_ = None if done else s_
            self.store_transition(z, s, a, r, s_, info)
            self.store_transition_online(z, s, a, r, s_, info)
            if done:
                s = env.reset(random_init=not self.fixed_init)
                z = self.sample_from_prior().view(1,-1).to(self.device)
            else:
                s = s_
        print(" --- Warmup Buffer Ends")


    def unpack_batch(self, batch):
        # `non_final_mask` is used for environments that have next state to be None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
            dtype=torch.bool).to(self.device)
        non_final_state_nxt = torch.FloatTensor([
            s for s in batch.s_ if s is not None]).to(self.device)
        latent = torch.cat(batch.z).to(self.device)  # stored as tensor
        state  = torch.FloatTensor(batch.s).to(self.device)
        action = torch.FloatTensor(batch.a).to(self.device).view(-1, self.actionDim)
        reward = torch.FloatTensor(batch.r).to(self.device)

        g_x = torch.FloatTensor(
            [info['g_x'] for info in batch.info]).to(self.device).view(-1)
        l_x = torch.FloatTensor(
            [info['l_x'] for info in batch.info]).to(self.device).view(-1)
        non_init_mask = torch.tensor(
            tuple(map(lambda s: not s['init'], batch.info)),
            dtype=torch.bool).view(-1).to(self.device)
        valid_mask = torch.logical_and(non_final_mask, non_init_mask)
        valid_state_next = torch.FloatTensor([
            s for s, t in zip(batch.s_, valid_mask) if t]).to(self.device)
        # non_init_mask = torch.tensor(
            # [~info['init'] for info in batch.info], dtype=torch.bool).to(self.device).view(-1)

        return non_final_mask, non_final_state_nxt, latent, state, action, reward, g_x, l_x, valid_mask, valid_state_next


    def update_critic(self, batch):

        non_final_mask, non_final_state_nxt, latent, state, action, reward, g_x, l_x, valid_mask, valid_state_next = self.unpack_batch(batch)
        self.critic.train()
        self.criticTarget.eval()
        self.actor.eval()
        self.disc.eval()

        #== augment reward with entropy for non final state
        valid_latent = latent[valid_mask]
        valid_state_stacked = torch.cat((state[valid_mask],
                                        valid_state_next), dim=1)
        log_prob_ps = self.disc(valid_state_stacked, valid_latent).sum(dim=1)
        log_prob_prior = self.latent_prior.log_prob(valid_latent).sum(dim=1)
        # print(reward[non_final_mask])
        # print(self.aug_reward_ratio*(log_prob_ps - log_prob_prior))
        # print(log_prob_ps - log_prob_prior)
        aug_cost = 1-(log_prob_ps - log_prob_prior).clamp(min=0.0, max=self.log_ps_bound)/self.log_ps_bound # normalized    #! the upper bound would depend on latent dimension
        # print(aug_cost)
        reward[valid_mask] -= self.aug_reward_range*aug_cost  #! nows it's -0.01 to 0.0
        # print(reward[valid_mask])
        non_final_latent = latent[non_final_mask]

        #== get Q(s,a) ==
        q1, q2 = self.critic(state, action, latent)  # Used to compute loss (non-target part).

        #== placeholder for target ==
        y = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        #== compute actor next_actions and feed to criticTarget ==
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(non_final_state_nxt, non_final_latent)
            next_q1, next_q2 = self.criticTarget(non_final_state_nxt, next_actions, non_final_latent)

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


    def update_disc(self):
        """
        Use detach_encoder=True to not update conv layers
        """
        batch = self.sample_batch(batch_size=self.disc_batch_size, online=True)
        _, _, latent, state, _, _, _, _, valid_mask, valid_state_next = self.unpack_batch(batch)
        if torch.sum(valid_mask) < 10:
            return 0

        self.disc.train()
        valid_state_stacked = torch.cat((state[valid_mask],
                                        valid_state_next), dim=1)
        log_prob_ps = self.disc(valid_state_stacked, latent[valid_mask], detach_encoder=self.share_disc_encoder_weight).sum(dim=1)
        loss_disc = -log_prob_ps.mean()

        #== backpropagation ==
        self.discOptimizer.zero_grad()
        loss_disc.backward()
        self.discOptimizer.step()
        return loss_disc.item()


    def update_actor(self, batch):
        """
        Use detach_encoder=True to not update conv layers
        """

        _, _, latent, state, _, _, _, _, _, _ = self.unpack_batch(batch)

        self.critic.eval()
        self.actor.train()
        self.disc.eval()

        action_sample, log_prob = self.actor.sample(state, latent, detach_encoder=True)
        q_pi_1, q_pi_2 = self.critic(state, action_sample, latent, detach_encoder=True)

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


    def sample_batch(self, batch_size=None, online=False):
        if online:
            sample = self.memory.sample_online
        else:
            sample = self.memory.sample
        if batch_size is None:
            transitions = sample(self.BATCH_SIZE)
        else:
            transitions = sample(batch_size)
        batch = TransitionLatent(*zip(*transitions))   
        return batch


    def update(self, timer):
        # if len(self.memory) < self.start_updates:
        #     return 0.0, 0.0, 0.0, 0.0

        #== EXPERIENCE REPLAY ==
        batch = self.sample_batch()

        self.critic.train()
        self.actor.train()
        self.disc.train()

        # Update critic
        loss_q = self.update_critic(batch)

        # Update actor and target critic
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % self.actor_freq == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            print('\r{:d}: (q, pi, ent, alpha) = ({:3.5f}/{:3.5f}/{:3.5f}/{:3.5f}).'.format(
                self.cntUpdate, loss_q, loss_pi, loss_entropy, loss_alpha), end=' ')

            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()
        self.disc.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha


    def learn(  self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=50,
                MAX_EVAL_EP_STEPS=100,
                warmupBuffer=True, warmupBufferRatio=1.0,
                warmupQ=False, warmupIter=10000,
                minStepBeforeOptimize=1000,
                optimizeFreq=100, numUpdatePerOptimize=128,
                curUpdates=None, checkPeriod=50000,
                plotFigure=True, storeFigure=False,
                showBool=False, vmin=-1, vmax=1, numRndTraj=200,
                storeModel=True, saveBest=False, outFolder='RA', verbose=True):

        useVis = 0
        if useVis:
            import visdom
            vis = visdom.Visdom(env='test_sac_maxent_dense_multi_obs_0', port=8098)
            q_loss_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Q Loss'))
            pi_loss_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Pi Loss'))
            disc_loss_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Disc Loss'))
            entropy_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Entropy'))
            success_window = vis.line(
                X=array([[0]]),
                Y=array([[0]]),
                opts=dict(xlabel='epoch', title='Success'))

        # == Build up networks
        self.build_network(verbose=verbose)
        self.setup_disc(verbose=verbose)
        print("Critic is using cuda: ", next(self.critic.parameters()).is_cuda)

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
            s = env.reset(random_init=not self.fixed_init)
            epCost = np.inf
            ep += 1

            # Sample z for episode
            z = self.latent_prior.sample().view(1,-1)

            # Rollout
            for _ in range(MAX_EP_STEPS):
                # Select action
                with torch.no_grad():
                    a, _ = self.actor.sample(
                        torch.from_numpy(s).float().to(self.device),
                        z.to(self.device))  # condition on latent
                    a = a.view(-1).cpu().numpy()

                # Interact with env
                s_, r, done, info = env.step(a)
                info['init'] = False  # indicate using z
                s_ = None if done else s_
                epCost = max(info["g_x"], min(epCost, info["l_x"]))

                # Store the transition in memory
                self.store_transition(z, s, a, r, s_, info)
                self.store_transition_online(z, s, a, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0 and self.cntUpdate > minStepBeforeOptimize:
                    self.actor.eval()
                    self.critic.eval()
                    policy = self.actor  # mean only and no std
                    def q_func(obs):
                        # TODO: for now use z=mean
                        obsTensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                        z = self.latent_mean.clone().view(1,-1)
                        u = self.actor(obsTensor, z).detach()
                        v = self.critic(obsTensor, u, z)[0].cpu().detach().numpy()[0]
                        return v

                    # Sample in some small region
                    x_range = [0.1, 0.3]
                    y_range = [-0.2, 0.2]
                    theta_range = [-np.pi/2, np.pi/2]
                    sample_x = np.random.uniform(x_range[0], x_range[1], (numRndTraj,1))
                    sample_y = np.random.uniform(y_range[0], y_range[1], (numRndTraj,1))
                    sample_theta = np.random.uniform(theta_range[0], theta_range[1], (numRndTraj,1))
                    sample_states = np.concatenate((sample_x, sample_y, sample_theta), axis=1)
                    results = env.simulate_trajectories(policy,
                        T=MAX_EVAL_EP_STEPS, states=sample_states, 
                        endType=endType,  latent_prior=self.latent_prior)[1]
                    # results = env.simulate_trajectories(policy,
                        # T=MAX_EVAL_EP_STEPS, num_rnd_traj=numRndTraj, 
                        # endType=endType, sample_inside_obs=False, 
                        # sample_inside_tar=False, 
                        # latent_prior=self.latent_prior)[1]
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
                            env.visualize(q_func, policy, vmin=0, boolPlot=True, latent_prior=self.latent_prior, normalize_v=True)
                        else:
                            env.visualize(q_func, policy, vmin=vmin, vmax=vmax, cmap='seismic', latent_prior=self.latent_prior, normalize_v=True)

                        if storeFigure:
                            figurePath = os.path.join(figureFolder,
                                '{:d}.png'.format(self.cntUpdate))
                            plt.savefig(figurePath)
                            plt.close()
                        if plotFigure:
                            plt.show()
                            plt.pause(0.001)
                            plt.close()

                # Count - do not update initially
                self.cntUpdate += 1
                if self.cntUpdate >= minStepBeforeOptimize and self.cntUpdate % optimizeFreq == 0:
                    loss_q, loss_pi, loss_entropy, loss_alpha = 0, 0, 0, 0

                    # Update discriminator - sample batches separately
                    loss_disc = 0
                    for timer in range(numUpdatePerOptimize*self.fit_freq):
                        loss_disc += self.update_disc()
                    loss_disc /= (numUpdatePerOptimize*self.fit_freq)

                    # Update critic/actor
                    for timer in range(numUpdatePerOptimize):
                        loss_q, loss_pi, loss_entropy, loss_alpha = self.update(timer)
                        trainingRecords.append([loss_q, loss_pi, loss_entropy, loss_alpha, loss_disc])

                        if (timer == numUpdatePerOptimize-1) and useVis:
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_q]]),
                                    win=q_loss_window,update='append')
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_pi]]),
                                    win=pi_loss_window,update='append')
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_entropy]]),
                                    win=entropy_window,update='append')
                            vis.line(X=array([[self.cntUpdate]]),
                                        Y=array([[loss_disc]]),
                                    win=disc_loss_window,update='append')

                    # Update gamma, lr etc.
                    self.updateHyperParam()
                    if self.cntUpdate % self.GAMMA_PERIOD == 0 and self.LEARN_ALPHA:
                        self.reset_alpha()

                    # Reset online buffer after updating
                    self.memory.reset_online()

                    # Increment fit_freq
                    self.fit_freq = math.ceil(self.fit_freq*self.fit_freq_rate)

                # Terminate early
                if done:
                    break

        endLearning = time.time()
        timeInitBuffer = 0
        timeInitQ = 0
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, '{:s}/model/'.format(outFolder))
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))

        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress
