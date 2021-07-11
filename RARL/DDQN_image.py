# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# Here we aim to minimize the cost. We make the following two modifications:
#  - a' = argmin_a' Q_policy(s', a')
#  - V(s') = Q_tar(s', a')
#  - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } } + (1-gamma) max{ g(s), l(s) }
#  - loss = E[ ( V(f(s,a)) - Q_policy(s,a) )^2 ]

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss
import torch.utils.data as Data

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy

from .neuralNetwork import ConvNet
from .DDQN import DDQN, Transition

class DDQN_image(DDQN):
    def __init__(self, CONFIG, actionSet, dimList, img_sz, kernel_sz, n_channel,
            mode='RA', terminalType='g', verbose=True):
        """
        __init__

        Args:
            CONFIG (object): configuration.
            actionSet (list): action set.
            dimList (np.ndarray): dimensions of each linear layer.
            img_sz (np.ndarray): image size of input.
            kernel_sz (np.ndarray): kernel size of each conv layer.
            n_channel (np.ndarray): number oof output channels of each conv layer.
            mode (str, optional): the learning mode. Defaults to 'RA'.
            terminalType (str, optional): terminal value. Defaults to 'g'.
            verbose (bool, optional): print or not. Defaults to True.
        """
        super(DDQN_image, self).__init__(CONFIG)

        self.mode = mode # 'normal' or 'RA' or 'safety'
        self.terminalType = terminalType

        #== ENV PARAM ==
        self.actionNum = len(actionSet)
        self.actionSet = actionSet

        #== Build NN for (D)DQN ==
        self.dimList = dimList
        self.img_sz = img_sz
        self.kernel_sz = kernel_sz
        self.n_channel = n_channel
        self.use_bn = CONFIG.USE_BN
        self.use_sm = CONFIG.USE_SM
        self.actType = CONFIG.ACTIVATION
        self.build_network(dimList, img_sz, kernel_sz, n_channel, self.actType,
                self.use_sm, self.use_bn, verbose)
        print("DDQN: mode-{}; terminalType-{}".format(self.mode, self.terminalType))


    def build_network(self, dimList, img_sz, kernel_sz, n_channel,
            actType='Tanh', use_sm=True, use_bn=False, verbose=True):
        """
        build_network [summary]

        Args:
            dimList (np.ndarray): dimensions of each linear layer.
            img_sz (np.ndarray): image size of input.
            kernel_sz (np.ndarray): kernel size of each conv layer.
            n_channel (np.ndarray): the first element is the input n_channel and the
                rest is the number of output channels of each conv layer.
            actType (str, optional): activation function. Defaults to 'Tanh'.
            use_sm (bool, optional): use spatial softmax or not. Defaults to True.
            use_bn (bool, optional): use batch normalization or not. Defaults to False.
            verbose (bool, optional): print or not. Defaults to True.
        """
        self.Q_network = ConvNet(   mlp_dimList=dimList,
                                    cnn_kernel_size=kernel_sz,
                                    input_n_channel=n_channel[0],
                                    output_n_channel=n_channel[1:],
                                    mlp_act=actType,
                                    mlp_output_act='Identity',
                                    img_size=img_sz,
                                    use_sm=use_sm,
                                    use_bn=use_bn,
                                    verbose=verbose)
        self.Q_network.to(self.device)
        self.target_network = copy.deepcopy(self.Q_network)
        print('Num of parameters in state encoder: %d' % sum(p.numel() for p in self.Q_network.parameters() if p.requires_grad))

        self.build_optimizer()


    def update(self):
        """
        update: update the critic.

        Returns:
            float: critic loss.
        """
        if len(self.memory) < self.BATCH_SIZE*20:
            return

        #== EXPERIENCE REPLAY ==
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x = \
            self.unpack_batch(batch)

        #== get Q(s,a) ==
        # `gather` reguires idx to be Long, input and index should have the same shape
        # with only difference at the dimension we want to extract value
        # out[i][j][k] = input[i][j][ index[i][j][k] ], which has the same dim as index
        # -> state_action_values = Q [ i ][ action[i] ]
        # view(-1): from mtx to vector
        self.Q_network.train()
        state_action_values = self.Q_network(state).gather(dim=1, index=action).view(-1)

        #== get a' by Q_policy: a' = argmin_a' Q_policy(s', a') ==
        with torch.no_grad():
            self.Q_network.eval()
            action_nxt = self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]

        #== get expected value ==
        state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)


        with torch.no_grad(): # V(s') = Q_tar(s', a'), a' is from Q_policy
            if self.double:
                self.target_network.eval()
                Q_expect = self.target_network(non_final_state_nxt)
            else:
                self.Q_network.eval()
                Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = Q_expect.gather(dim=1, index=action_nxt).view(-1)

        #== Discounted Reach-Avoid Bellman Equation (DRABE) ==
        if self.mode == 'RA':
            expected_state_action_values = torch.zeros(self.BATCH_SIZE).float().to(self.device)
            # V(s) = max{ g(s), min{ l(s), V(s') } }, Q(s, u) = V( f(s,u) )
            non_terminal = torch.max(
                g_x[non_final_mask],
                torch.min(
                    l_x[non_final_mask],
                    state_value_nxt[non_final_mask]
                )
            )
            terminal = torch.max(l_x, g_x)

            # normal state
            expected_state_action_values[non_final_mask] = \
                non_terminal * self.GAMMA + \
                terminal[non_final_mask] * (1-self.GAMMA)

            # terminal state
            final_mask = torch.logical_not(non_final_mask)
            if self.terminalType == 'g':
                expected_state_action_values[final_mask] = g_x[final_mask]
            elif self.terminalType == 'max':
                expected_state_action_values[final_mask] = terminal[final_mask]
            else:
                raise ValueError("invalid terminalType")
        elif self.mode == 'safety':
            # V(s) = max{ g(s), V(s') }, Q(s, u) = V( f(s,u) )
            expected_state_action_values = torch.zeros(self.BATCH_SIZE).float().to(self.device)
            non_terminal = torch.max(
                g_x[non_final_mask],
                state_value_nxt[non_final_mask]
            )
            terminal = g_x[non_final_mask]

            # normal state
            expected_state_action_values[non_final_mask] = \
                non_terminal * self.GAMMA + \
                terminal * (1-self.GAMMA)

            # terminal state
            final_mask = torch.logical_not(non_final_mask)
            expected_state_action_values[final_mask] = g_x[final_mask]
        else: # V(s) = -r(s, a) + gamma * V(s')
            expected_state_action_values = state_value_nxt * self.GAMMA - reward

        #== regression: Q(s, a) <- V(s) ==
        loss = smooth_l1_loss(input=state_action_values, target=expected_state_action_values.detach())

        #== backpropagation ==
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_target_network()

        return loss.item()


    def initBuffer(self, env):
        """
        initBuffer: randomly put some transitions into the memory replay buffer.

        Args:
            env (gym.Env Obj.): environment.
        """
        cnt = 0
        while len(self.memory) < self.memory.capacity:
            cnt += 1
            print('\rWarmup Buffer [{:d}]'.format(cnt), end='')
            s = env.reset()
            a, a_idx = self.select_action(s, explore=True)
            s_, r, done, info = env.step(a_idx)
            s_ = None if done else s_
            self.store_transition(s, a_idx, r, s_, info)
        print(" --- Warmup Buffer Ends")


    def initQ(  self, env, warmupIter, outFolder, num_warmup_samples=200, batch_sz=64,
                vmin=-1, vmax=1, plotFigure=True, storeFigure=True):
        """
        initQ: initalize Q-network.

        Args:
            env (gym.Env Obj.): environment.
            warmupIter (int, optional): the number of iterations in the
                Q-network warmup.
            outFolder (str, optional): the relative folder path with respect to
                model/ and figure/.
            num_warmup_samples (int, optional): Defaults to 200.
            vmin (float, optional): the minimal value in the colorbar. Defaults to -1.
            vmax (float, optional): the maximal value in the colorbar. Defaults to 1.
            plotFigure (bool, optional): plot figures if True. Defaults to True.
            storeFigure (bool, optional): store figures if True. Defaults to False.

        Returns:
            np.ndarray: loss of fitting Q-values to heuristic values.
        """
        lossArray = []
        states, value = env.get_warmup_examples(num_warmup_samples)
        states = torch.FloatTensor(states).to(self.device)
        value = torch.FloatTensor(value).to(self.device)
        heuristic_dataset = Data.TensorDataset(states, value)
        heuristic_dataloader = Data.DataLoader( heuristic_dataset,
                                                batch_size=batch_sz,
                                                shuffle=True)
        self.Q_network.train()
        startInitQ = time.time()
        for ep_tmp in range(warmupIter):
            i = 0
            lossList = []
            for stateTensor, valueTensor in heuristic_dataloader:
                i += 1
                v = self.Q_network(stateTensor)
                v = torch.mean(v, dim=1, keepdim=True)
                # valueTensor = valueTensor.repeat(1, 3)
                loss = mse_loss(input=v, target=valueTensor, reduction='sum')

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                lossList.append(loss.detach().cpu().numpy())
                print('\rWarmup Q [{:d}-{:d}]. MSE = {:f}'.format(
                    ep_tmp+1, i, loss.detach()), end='')
            lossArray.append(lossList)
        endInitQ = time.time()
        timeInitQ = endInitQ - startInitQ
        print(" --- Warmup Q Ends after {:.1f} seconds".format(timeInitQ))

        self.target_network.load_state_dict(self.Q_network.state_dict()) # hard replace
        self.build_optimizer()
        modelFolder = os.path.join(outFolder, 'model')
        os.makedirs(modelFolder, exist_ok=True)
        self.save('init', modelFolder)

        if plotFigure or storeFigure:
            self.Q_network.eval()
            env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap='seismic')
            if storeFigure:
                figureFolder = os.path.join(outFolder, 'figure')
                os.makedirs(figureFolder, exist_ok=True)
                figurePath = os.path.join(figureFolder, 'initQ.png')
                plt.savefig(figurePath)
            if plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()

        return np.array(lossArray)


    def learn(self, env, warmupBuffer=True, warmupQ=False, warmupIter=1000,
            MAX_UPDATES=400000, curUpdates=None, MAX_EP_STEPS=250,
            runningCostThr=None, checkPeriod=50000, verbose=True,
            plotFigure=True, showBool=False, vmin=-1, vmax=1, numRndTraj=200,
            outFolder='RA', storeFigure=False, storeModel=True, storeBest=False):
        """
        learn: Learns the vlaue function.

        Args:
            env (gym.Env Obj.): environment.
            warmupBuffer (bool, optional): fill the replay buffer if True.
                Defaults to True.
            warmupQ (bool, optional): train the Q-network by (l_x, g_x) if True.
                Defaults to False.
            warmupIter (int, optional): the number of iterations in the
                Q-network warmup. Defaults to 1000.
            MAX_UPDATES (int, optional): the maximal number of gradient updates.
                Defaults to 400000.
            curUpdates (int, optional): set the current number of updates
                (usually used when restoring trained models). Defaults to None.
            MAX_EP_STEPS (int, optional): the number of steps in an episode.
                Defaults to 250.
            runningCostThr (float, optional): ends the training if the running
                cost is smaller than the threshold. Defaults to None.
            checkPeriod (int, optional): the period we check the performance.
                Defaults to 50000.
            verbose (bool, optional): output message if True. Defaults to True.
            plotFigure (bool, optional): plot figures if True. Defaults to True.
            showBool (bool, optional): use bool value if True. Defaults to False.
            vmin (float, optional): the minimal value in the colorbar. Defaults to -1.
            vmax (float, optional): the maximal value in the colorbar. Defaults to 1.
            numRndTraj (int, optional): the number of random trajectories used
                to obtain the success ratio. Defaults to 200.
            outFolder (str, optional): the parent folder of model/ and figure/.
                Defaults to 'RA'.
            storeFigure (bool, optional): store figures if True. Defaults to False.
            storeModel (bool, optional): store models if True. Defaults to True.
            storeBest (bool, optional): only store the best model if True.
                Defaults to False.

        Returns:
            trainingRecords (np.ndarray): loss_critic for every update.
            trainProgress (np.ndarray): each entry consists of the
                success/failure/unfinished ratio of random trajectories and is
                checked periodically.
        """

        # == Warmup Buffer ==
        startInitBuffer = time.time()
        if warmupBuffer:
            self.initBuffer(env)
        endInitBuffer = time.time()

        # == Warmup Q ==
        startInitQ = time.time()
        if warmupQ:
            self.initQ(env, warmupIter=warmupIter, outFolder=outFolder,
                    plotFigure=plotFigure, storeFigure=storeFigure,
                    vmin=vmin, vmax=vmax)
        endInitQ = time.time()

        # == Main Training ==
        startLearning = time.time()
        trainingRecords = []
        runningCost = 0.
        trainProgress = []
        checkPointSucc = 0.
        ep = 0

        if curUpdates is not None:
            self.cntUpdate = curUpdates
            print("starting from {:d} updates".format(self.cntUpdate))

        if storeModel:
            modelFolder = os.path.join(outFolder, 'model')
            os.makedirs(modelFolder, exist_ok=True)
        if storeFigure:
            figureFolder = os.path.join(outFolder, 'figure')
            os.makedirs(figureFolder, exist_ok=True)

        if self.mode=='safety':
            endType = 'fail'
        else:
            endType = 'TF'

        while self.cntUpdate <= MAX_UPDATES:
            s = env.reset()
            epCost = 0.
            ep += 1
            # Rollout
            for step_num in range(MAX_EP_STEPS):
                # Select action
                a, a_idx = self.select_action(s, explore=True)

                # Interact with env
                s_, r, done, info = env.step(a_idx)
                s_ = None if done else s_

                # Rollout record
                epCost += r
                _state = info['state']
                g_x = env.safety_margin(_state)
                l_x = env.target_margin(_state)
                if step_num == 0:
                    maxG = g_x
                    current = max(l_x, maxG)
                    minV = current
                else:
                    maxG = max(maxG, g_x)
                    current = max(l_x, maxG)
                    minV = min(current, minV)

                # Store the transition in memory
                self.store_transition(s, a_idx, r, s_, info)
                s = s_

                # Check after fixed number of gradient updates
                if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
                    policy = lambda obs: self.select_action(obs, explore=False)[1]
                    results = env.simulate_trajectories(policy,
                        T=MAX_EP_STEPS, num_rnd_traj=numRndTraj, endType=endType,
                        sample_inside_obs=False, sample_inside_tar=False)[1]
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
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        print('\nAfter [{:d}] updates:'.format(self.cntUpdate))
                        print('  - eps={:.2f}, gamma={:.4f}, lr={:.1e}.'.format(
                            self.EPSILON, self.GAMMA, lr))
                        if self.mode == 'safety':
                            print('  - success/failure ratio:', end=' ')
                        else:
                            print('  - success/failure/unfinished ratio:', end=' ')
                        with np.printoptions(formatter={'float': '{: .2f}'.format}):
                            print(np.array(trainProgress[-1]))

                    if storeModel:
                        if storeBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self.save(self.cntUpdate, modelFolder)
                        else:
                            self.save(self.cntUpdate, modelFolder)

                    if plotFigure or storeFigure:
                        self.Q_network.eval()
                        if showBool:
                            env.visualize(self.Q_network, policy, self.device,
                                vmin=0, boolPlot=True)
                        else:
                            normalize_v = not (self.mode == 'RA' or self.mode == 'safety')
                            env.visualize(self.Q_network, policy, self.device,
                                vmin=vmin, vmax=vmax, cmap='seismic', normalize_v=normalize_v)
                        if storeFigure:
                            figurePath = os.path.join(figureFolder,
                                '{:d}.png'.format(self.cntUpdate))
                            plt.savefig(figurePath)
                        if plotFigure:
                            plt.show()
                            plt.pause(0.001)
                        plt.close()

                # Perform one step of the optimization (on the target network)
                lossC = self.update()
                trainingRecords.append(lossC)
                self.cntUpdate += 1
                self.updateHyperParam()

                # Terminate early
                if done:
                    break

            # Rollout report
            runningCost = runningCost * 0.9 + epCost * 0.1
            if verbose:
                print('\r[{:d}-{:d}]: reach-avoid/lagrange cost'.format(
                    ep, self.cntUpdate), end=' ')
                print('= ({:3.2f}/{:.2f}) after {:d} steps.'.format(
                    minV, epCost, step_num+1), end='')

            # Check stopping criteria
            if runningCostThr != None:
                if runningCost <= runningCostThr:
                    print("\n At Updates[{:3.0f}] Solved! Running cost is now {:3.2f}!".format(self.cntUpdate, runningCost))
                    env.close()
                    break
        endLearning = time.time()
        timeInitBuffer = endInitBuffer - startInitBuffer
        timeInitQ = endInitQ - startInitQ
        timeLearning = endLearning - startLearning
        self.save(self.cntUpdate, modelFolder)
        print('\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}'.format(
            timeInitBuffer, timeInitQ, timeLearning))
        trainingRecords = np.array(trainingRecords)
        trainProgress = np.array(trainProgress)
        return trainingRecords, trainProgress


    def select_action(self, state, explore=False):
        """
        select_action

        Args:
            state (np.ndarray): state
            explore (bool, optional): randomized the deterministic action by
                epsilon-greedy. Defaults to False.

        Returns:
            np.ndarray: action
            int: action index
        """
        self.Q_network.eval()
        # tensor.min() returns (value, indices), which are in tensor form
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if (np.random.rand() < self.EPSILON) and explore:
            action_index = np.random.randint(0, self.actionNum)
        else:
            action_index = self.Q_network(state).min(dim=1)[1].item()
        return self.actionSet[action_index], action_index