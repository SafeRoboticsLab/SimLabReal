# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.distributions import Normal
import copy

from .neuralNetwork import MLP, ConvNet


#== Critic ==
class TwinnedQNetwork(nn.Module):
    def __init__(self, dimList, actType='Tanh', device='cpu', image=False,
            actionDim=1, verbose=True):

        super(TwinnedQNetwork, self).__init__()
        self.image = image
        if verbose:
            print("The neural networks for CRITIC have the architecture as below:")
        if image:
            self.Q1 = ConvNet(  mlp_dimList=dimList,
                                mlp_append_dim=actionDim,
                                mlp_act=actType,
                                mlp_output_act='Identity',
                                verbose=verbose).to(device)
            # self.Q2 = ConvNet(  mlp_dimList=dimList,
            #                     mlp_append_dim=actionDim,
            #                     mlp_act=actType,
            #                     mlp_output_act='Identity',
            #                     verbose=False).to(device)
        else:
            self.Q1 = MLP(dimList, actType, verbose=verbose).to(device)
            # self.Q2 = MLP(dimList, actType, verbose=False).to(device)
        self.Q2 = copy.deepcopy(self.Q1)
        self.device = device

    def forward(self, states, actions):
        if self.image:
            states = states.to(self.device)
            actions = actions.to(self.device)
            q1 = self.Q1(x=states, mlp_append=actions)
            q2 = self.Q2(x=states, mlp_append=actions)
        else:
            x = torch.cat([states, actions], dim=-1).to(self.device)
            q1 = self.Q1(x)
            q2 = self.Q2(x)
        return q1, q2


#== Policy (Actor) Model ==
class GaussianPolicy(nn.Module):
    def __init__(self, dimList, actionSpace, actType='Tanh', device='cpu', verbose=True):
        super(GaussianPolicy, self).__init__()
        self.device = device
        if verbose:
            print("The neural network for MEAN has the architecture as below:")
        self.mean = MLP(dimList, actType, output_activation=nn.Tanh,
            verbose=verbose).to(device)
        if verbose:
            print("The neural network for LOG_STD has the architecture as below:")
        self.log_std = MLP(dimList, actType, output_activation=nn.Identity,
            verbose=verbose).to(device)

        self.actionSpace = actionSpace
        self.a_max = self.actionSpace.high[0]
        self.a_min = self.actionSpace.low[0]
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0

        self.LOG_STD_MAX = 1
        self.LOG_STD_MIN = -10
        # self.log_scale = (self.LOG_STD_MAX - self.LOG_STD_MIN) / 2.0
        # self.log_bias = (self.LOG_STD_MAX + self.LOG_STD_MIN) / 2.0
        self.eps = 1e-8


    # def forward(self, state):
    #     stateTensor = state.to(self.device)
    #     mean = self.mean(stateTensor)
    #     log_std = self.log_std(stateTensor)
    #     log_std = torch.tanh(log_std) * self.log_scale + self.log_bias
    #     return mean, log_std


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        return mean * self.scale + self.bias


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        log_std = self.log_std(stateTensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        std = torch.exp(log_std)
        normalRV = Normal(mean, std)

        x = normalRV.rsample()  # reparameterization trick (mean + std * N(0,1))
        y = torch.tanh(x)   # constrain the output to be within [-1, 1]

        action = y * self.scale + self.bias
        log_prob = normalRV.log_prob(x)

        # Get the correct probability: x -> a, a = c * y + b, y = tanh x
        # followed by: p(a) = p(x) x |det(da/dx)|^-1
        # log p(a) = log p(x) - log |det(da/dx)|
        # log |det(da/dx)| = sum log (d a_i / d x_i)
        # d a_i / d x_i = c * ( 1 - y_i^2 )
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
        # log_prob = log_prob.sum(1, keepdim=True)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum()
        # mean = torch.tanh(mean) * self.scale + self.bias
        return action, log_prob


class DeterministicPolicy(nn.Module):
    def __init__(self, dimList, actionSpace, actType='Tanh', device='cpu',
        noiseStd=0.1, noiseClamp=0.5, image=False, verbose=True):

        super(DeterministicPolicy, self).__init__()
        self.device = device
        if verbose:
            print("The neural network for ACTOR-MEAN has the architecture as below:")
        if image:
            self.mean = ConvNet(mlp_dimList=dimList,
                                mlp_act=actType,
                                mlp_output_act='Tanh',
                                verbose=verbose).to(device)
        else:
            self.mean = MLP(dimList, 
                            actType, 
                            output_activation=nn.Tanh,
                            verbose=verbose).to(device)
        # self.noise = Normal(0., noiseStd)
        self.noiseClamp = noiseClamp
        self.actionSpace = actionSpace
        self.noiseStd = noiseStd

        self.a_max = self.actionSpace.high[0]
        self.a_min = self.actionSpace.low[0]
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0


    def forward(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        return mean * self.scale + self.bias


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean = self.forward(stateTensor)
        # noise = self.noise.sample().to(self.device)
        noise = torch.randn_like(mean) * self.noiseStd
        noise_clipped = torch.clamp(noise, -self.noiseClamp, self.noiseClamp)

        # Action.
        action = mean + noise
        action = torch.clamp(action, self.a_min, self.a_max)

        # Target action.
        action_target = mean + noise_clipped
        action_target = torch.clamp(action_target, self.a_min, self.a_max)

        return action, action_target