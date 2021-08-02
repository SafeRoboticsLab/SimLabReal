# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import torch
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import os
import time
import math

from .SAC_image_maxEnt import SAC_image_maxEnt


class SAC_image_maxEnt_vec(SAC_image_maxEnt):
    def __init__(self, CONFIG, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """
        super(SAC_image_maxEnt_vec, self).__init__(CONFIG, verbose)


    def initBuffer(self, venv, ratio=1.0):
        cnt = 0
        s = venv.reset(random_init=not self.fixed_init)
        z = self.sample_from_prior().view(1,-1).to(self.device) # doesn't matter
        while len(self.memory) < self.memory.capacity * ratio:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            a_all = torch.tensor([venv.action_space.sample() for _ in range(self.n_envs)])
            s_all, r_all, done_all, info_all = venv.step(a_all)
            
            # TODO: save transitions in batch?
            for env_ind, (s_, r, done, info) in enumerate(zip(s_all, r_all, done_all, info_all)):
                info['init'] = True #! indicate not using z
                # s_ = None if done else s_
                self.store_transition(z, s[env_ind].unsqueeze(0), a_all[env_ind].unsqueeze(0), r, s_.unsqueeze(0), done, info)
                self.store_transition_online(z, s[env_ind].unsqueeze(0), a_all[env_ind].unsqueeze(0), r, s_.unsqueeze(0), done, info)
                s[env_ind] = s_
        print(" --- Warmup Buffer Ends")


    def learn(  self, venv, env, MAX_UPDATES=2000000, 
                #   MAX_EP_STEPS=50,MAX_EVAL_EP_STEPS=100,
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
            vis = visdom.Visdom(env='test_sac_maxent_dense_multi_obs_2', port=8098)
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

        # Save number of environments - too lazy to use config
        self.n_envs = len(venv.remotes)

        # == Build up networks
        self.build_network(verbose=verbose)
        self.setup_disc(verbose=verbose)
        print("Critic is using cuda: ", next(self.critic.parameters()).is_cuda)

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(venv, ratio=warmupBufferRatio)
        endInitBuffer = time.time()
        print('Time to warm up: ', endInitBuffer-startInitBuffer)

        # == Warmup Q ==    # Not implemented for vecEnvs
        startInitQ = time.time()
        if warmupQ:
            self.initQ(venv, warmupIter=warmupIter, outFolder=outFolder,
                vmin=vmin, vmax=vmax, plotFigure=plotFigure,
                storeFigure=storeFigure)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainingRecords = []
        trainProgress = []
        checkPointSucc = 0.
        checkPeriodCount = 0
        optimizePeriodCount = 0

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

        # Reset all envs
        s = venv.reset(random_init=not self.fixed_init)
        z = self.latent_prior.sample((self.n_envs,)).to(self.device)
        while self.cntUpdate <= MAX_UPDATES:

            # Set train modes for all envs
            venv.env_method('set_train_mode')

            # Select action
            with torch.no_grad():
                a_all, _ = self.actor.sample(s, z)  # condition on latent

            # Interact with env
            s_all, r_all, done_all, info_all = venv.step(a_all)
            for env_ind, (s_, r, done, info) in enumerate(zip(s_all, r_all, done_all, info_all)):
                info['init'] = False  # indicate using z
                # s_ = None if done else s_

                # Store the transition in memory
                self.store_transition(z[env_ind].unsqueeze(0), s[env_ind].unsqueeze(0), a_all[env_ind].unsqueeze(0), r, s_.unsqueeze(0), done, info)
                self.store_transition_online(z[env_ind].unsqueeze(0), s[env_ind].unsqueeze(0), a_all[env_ind].unsqueeze(0), r, s_.unsqueeze(0), done, info)
                s[env_ind] = s_

                # Resample z
                if done:
                    z[env_ind] = self.latent_prior.sample().to(self.device)

            # Train policies
            if self.cntUpdate >= minStepBeforeOptimize and \
                optimizePeriodCount >= optimizeFreq:
                optimizePeriodCount = 0
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

            # Check after fixed number of gradient updates
            if self.cntUpdate >= minStepBeforeOptimize and \
                checkPeriodCount >= checkPeriod:
                checkPeriodCount = 0

                # Set eval modes for all envs
                venv.env_method('set_eval_mode')
                
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
                results = venv.simulate_trajectories(policy,states=sample_states, endType=endType, latent_prior=self.latent_prior)
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
                    if showBool:    #! using only a single env for visualizing, also can't use vector envs as q_func cannot be pickled
                        env.visualize(q_func=q_func, policy=policy, vmin=0, boolPlot=True, latent_prior=self.latent_prior, normalize_v=True)
                    else:
                        env.visualize(q_func=q_func, policy=policy, vmin=vmin, vmax=vmax, cmap='seismic', latent_prior=self.latent_prior, normalize_v=True)

                    if storeFigure:
                        figurePath = os.path.join(figureFolder,
                            '{:d}.png'.format(self.cntUpdate))
                        plt.savefig(figurePath)
                        plt.close()
                    if plotFigure:
                        plt.show()
                        plt.pause(0.001)
                        plt.close()

            # Count
            self.cntUpdate += self.n_envs
            checkPeriodCount += self.n_envs
            optimizePeriodCount += self.n_envs

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
