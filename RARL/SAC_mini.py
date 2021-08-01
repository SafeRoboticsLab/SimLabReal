# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
import os

from .model import SACPiNetwork, SACTwinnedQNetwork
from .scheduler import StepLRMargin
from .utils import soft_update, save_model
import copy
import pickle

class SAC_mini(object):
    def __init__(self, CONFIG, CONFIG_ARCH, verbose=True):
        """
        __init__: initialization.

        Args:
            CONFIG (Class object): hyper-parameter configuration.
            verbose (bool, optional): print info or not. Defaults to True.
        """

        self.saved = False
        self.CONFIG = CONFIG
        self.CONFIG_ARCH = CONFIG_ARCH

        #== ENV PARAM ==
        self.obsChannel = CONFIG.OBS_CHANNEL
        self.actionMag = CONFIG.ACTION_MAG
        self.actionDim = CONFIG.ACTION_DIM
        self.img_sz = CONFIG.IMG_SZ

        #== PARAM ==
        # Learning Rate
        self.LR_C = CONFIG.LR_C
        self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
        self.LR_C_DECAY = CONFIG.LR_C_DECAY
        self.LR_C_END = CONFIG.LR_C_END
        self.LR_A = CONFIG.LR_A
        self.LR_A_PERIOD = CONFIG.LR_A_PERIOD
        self.LR_A_DECAY = CONFIG.LR_A_DECAY
        self.LR_A_END = CONFIG.LR_A_END

        # NN: batch size, maximal number of NNs stored
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.MAX_MODEL = CONFIG.MAX_MODEL
        self.device = CONFIG.DEVICE

        # Discount Factor
        self.GammaScheduler = StepLRMargin( initValue=CONFIG.GAMMA,
            period=CONFIG.GAMMA_PERIOD, decay=CONFIG.GAMMA_DECAY,
            endValue=CONFIG.GAMMA_END, goalValue=1.)
        self.GAMMA = self.GammaScheduler.get_variable()

        # Target Network Update
        self.TAU = CONFIG.TAU

        # alpha-related hyper-parameters
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

        # reach-avoid setting
        self.mode = CONFIG.MODE
        self.terminalType = CONFIG.TERMINAL_TYPE

        self.build_network(CONFIG_ARCH, verbose=verbose)


    def build_network(self, CONFIG, verbose=True):
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


    def reset_alpha(self):
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alphaOptimizer = Adam([self.log_alpha], lr=self.LR_Al)
        self.log_alphaScheduler = lr_scheduler.StepLR(self.log_alphaOptimizer,
            step_size=self.LR_Al_PERIOD, gamma=self.LR_Al_DECAY)


    @property
    def alpha(self):
        return self.log_alpha.exp()


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


    def update(self, batch, timer, update_period=2):
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha


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


    def save(self, step, logs_path):
        path_c = os.path.join(logs_path, 'critic')
        path_a = os.path.join(logs_path, 'actor')
        save_model(self.critic, step, path_c, 'critic', self.MAX_MODEL)
        save_model(self.actor,  step, path_a, 'actor',  self.MAX_MODEL)
        if not self.saved:
            config_path = os.path.join(logs_path, "CONFIG.pkl")
            pickle.dump(self.CONFIG, open(config_path, "wb"))
            config_path = os.path.join(logs_path, "CONFIG_ARCH.pkl")
            pickle.dump(self.CONFIG_ARCH, open(config_path, "wb"))
            self.saved = True


    def check(self, env, cntStep, MAX_EVAL_EP_STEPS, numRndTraj, verbose=True):
        if self.mode=='safety':
            endType = 'fail'
        else:
            endType = 'TF'

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
            trainProgress = np.array([success, failure])
        else:
            success  = np.sum(results==1) / results.shape[0]
            failure  = np.sum(results==-1)/ results.shape[0]
            unfinish = np.sum(results==0) / results.shape[0]
            trainProgress = np.array([success, failure, unfinish])

        if verbose:
            lr = self.actorOptimizer.state_dict()['param_groups'][0]['lr']
            print('\n{} policy after [{}] steps:'.format(self.mode, cntStep))
            print('  - gamma={:.6f}, lr={:.1e}, alpha={:.1e}.'.format(
                self.GAMMA, lr, self.alpha))
            if self.mode == 'safety':
                print('  - success/failure ratio:', end=' ')
            else:
                print('  - success/failure/unfinished ratio:', end=' ')
            with np.printoptions(formatter={'float': '{: .2f}'.format}):
                print(trainProgress)
        self.actor.train()
        self.critic.train()

        return trainProgress