# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
from torch.distributions import Normal
import copy
import numpy as np
from collections import OrderedDict

from .neuralNetwork import MLP, ConvNet, activationDict

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder(torch.nn.Module):
    """Conv layers shared by actor and critic in SAC."""
    def __init__(self,  input_n_channel,
                        img_sz, 
                        kernel_sz,
                        n_channel,
                        use_sm=True,
                        device='cpu',
                        verbose=True,
                ):
        super().__init__()
        if verbose:
            print("The neural network for encoder has the architecture as below:")
        self.conv = ConvNet(input_n_channel=input_n_channel,
                            mlp_dimList=[], # no linear layers
                            cnn_kernel_size=kernel_sz,
                            output_n_channel=n_channel,
                            img_size=img_sz,
                            use_sm=use_sm,
                            use_bn=False,
                            verbose=verbose).to(device)

    def forward(self, image, detach=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        out = self.conv(image)
        if detach:
            out = out.detach()
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers - assume actor and critic have same conv structure"""
        for module_ind, module in enumerate(self.conv.moduleList):
            for layer_ind, layer in enumerate(module):
                if isinstance(layer, nn.Conv2d):
                    # print(layer)
                    tie_weights(src=source.conv.moduleList[module_ind][layer_ind], trg=layer)


class SACPiNetwork(torch.nn.Module):
    def __init__(self, input_n_channel,
                        mlp_dim, 
                        actionDim,
                        actionMag,
                        actType,    # for MLP; ReLU default for conv
                        img_sz, 
                        kernel_sz,
                        n_channel,
                        use_sm=True,
                        device='cpu',
                        verbose=True,
                        ):
        super().__init__()
        self.device = device
    
        # Conv layers shared with critic
        self.encoder = Encoder(input_n_channel,
                                img_sz,
                                kernel_sz,
                                n_channel,
                                use_sm,
                                device,
                                False)
        if use_sm:
            dim_conv_out = n_channel[-1]*2 # assume spatial softmax
        else:
            dim_conv_out = n_channel[-1]*img_sz ** 2
        mlp_dim = [dim_conv_out] + mlp_dim + [actionDim]

        # Linear layers
        self.mlp = GaussianPolicy(mlp_dim, actionMag, actType, device, verbose)


    def forward(self, image, detach_encoder=False):
        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().to(self.device)
            np_input = True

        # Make batch
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            B = 1
        else:
            B = image.shape[0]

        # Forward pass
        conv_out = self.encoder.forward(image, detach=detach_encoder)
        output = self.mlp(conv_out.view(B, -1))

        # Restore dimension
        if B == 1:
            output = output.squeeze(0)

        # Convert back to np
        if np_input:
            output = output.detach().cpu().numpy()

        return output


    def sample(self, image, detach_encoder=False):
        conv_out = self.encoder.forward(image, detach=detach_encoder)
        output = self.mlp.sample(conv_out)
        return output


class SACTwinnedQNetwork(torch.nn.Module):
    def __init__(self,  input_n_channel,
                        mlp_dim, 
                        actionDim,
                        actType,    # for MLP; ReLU default for conv
                        img_sz, 
                        kernel_sz,
                        n_channel,
                        use_sm=True,
                        device='cpu',
                        verbose=True,
                        ):

        super().__init__()
        self.device = device

        # Conv layers shared with critic
        self.encoder = Encoder(input_n_channel,
                                img_sz,
                                kernel_sz,
                                n_channel,
                                use_sm,
                                device,
                                verbose)
        if use_sm:
            dim_conv_out = n_channel[-1]*2 # assume spatial softmax
        else:
            dim_conv_out = n_channel[-1]*img_sz ** 2
        mlp_dim = [dim_conv_out+actionDim] + mlp_dim + [1]

        # Double critics
        self.Q1 = nn.Sequential( OrderedDict([
                    ('linear_1',    nn.Linear(mlp_dim[0], mlp_dim[1])),
                    ('norm_1',      nn.LayerNorm(mlp_dim[1])),
                    ('linear_2',    nn.Linear(mlp_dim[1], mlp_dim[2])),
                    ('act_2',       activationDict[actType]),
                    ('linear_3',    nn.Linear(mlp_dim[2], mlp_dim[3])),
        ])).to(device)
        self.Q2 = copy.deepcopy(self.Q1)
        if verbose:
            print("The MLP for critic has the architecture as below:")
            print(self.Q1)


    def forward(self, image, actions, detach_encoder=False):

        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().to(self.device)
            np_input = True

        # Make batch
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            B = 1
        else:
            B = image.shape[0]

        # Forward pass
        conv_out = self.encoder.forward(image, detach=detach_encoder)
        q_input = torch.cat([conv_out.view(B, -1), 
                            actions.view(B, -1)], dim=1)
        q1 = self.Q1(q_input)
        q2 = self.Q2(q_input)

        # Restore dimension
        if B == 1:
            q1 = q1.squeeze(0)
            q2 = q2.squeeze(0)

        # Convert back to np
        if np_input:
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()

        return q1, q2


#== Critic ==
class TwinnedQNetwork(nn.Module):
    def __init__(self, dimList, img_sz, actType='Tanh', device='cpu', image=False, actionDim=1, verbose=True, **kwargs):

        super(TwinnedQNetwork, self).__init__()
        self.image = image
        if verbose:
            print("The neural networks for CRITIC have the architecture as below:")
        if image:
            kernel_sz = kwargs.get('kernel_sz')
            n_channel = kwargs.get('n_channel')
            self.Q1 = ConvNet(  mlp_dimList=dimList,
                                cnn_kernel_size=kernel_sz,
                                cnn_channel_numbers=n_channel,
                                mlp_append_dim=actionDim,
                                mlp_act=actType,
                                mlp_output_act='Identity',
                                img_size=img_sz,
                                verbose=verbose).to(device)
        else:
            self.Q1 = MLP(dimList, actType, verbose=verbose).to(device)
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
    def __init__(self, dimList, actionMag, actType='Tanh', device='cpu', verbose=True):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.mean = nn.Sequential( OrderedDict([
                    ('linear_1',    nn.Linear(dimList[0], dimList[1])),
                    ('norm_1',      nn.LayerNorm(dimList[1])),
                    ('linear_2',    nn.Linear(dimList[1], dimList[2])),
                    ('act_2',       activationDict[actType]),
                    ('linear_3',    nn.Linear(dimList[2], dimList[3])),
                    ('act_3',       nn.Tanh()),
        ])).to(device)
        # self.mean = MLP(dimList, actType, output_activation=nn.Tanh,
        #     verbose=verbose).to(device)
        self.log_std = nn.Sequential( OrderedDict([
                    ('linear_1',    nn.Linear(dimList[0], dimList[1])),
                    ('norm_1',      nn.LayerNorm(dimList[1])),
                    ('linear_2',    nn.Linear(dimList[1], dimList[2])),
                    ('act_2',       activationDict[actType]),
                    ('linear_3',    nn.Linear(dimList[2], dimList[3])),
        ])).to(device)
        # self.log_std = MLP(dimList, actType, output_activation=nn.Identity,
        #     verbose=verbose).to(device)
        if verbose:
            print("The MLP for MEAN has the architecture as below:")
            print(self.mean)
            print("The MLP for LOG_STD has the architecture as below:")
            print(self.log_std)

        self.a_max = actionMag
        self.a_min = -actionMag
        self.scale = (self.a_max - self.a_min) / 2.0    # basically the mag
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


    def forward(self, state):   # mean only
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        return mean * self.scale + self.bias


    def sample(self, state):
        stateTensor = state.to(self.device)
        mean = self.mean(stateTensor)
        log_std = self.log_std(stateTensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Sample
        normalRV = Normal(mean, std)
        x = normalRV.rsample()  # reparameterization trick (mean + std * N(0,1))
        log_prob = normalRV.log_prob(x)

        # Get action
        y = torch.tanh(x)   # constrain the output to be within [-1, 1]
        action = y * self.scale + self.bias

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
    def __init__(self, dimList, img_sz, actionSpace, actType='Tanh', device='cpu',
            noiseStd=0.1, noiseClamp=0.5, image=False, verbose=True, **kwargs):

        super(DeterministicPolicy, self).__init__()
        self.device = device
        if verbose:
            print("The neural network for ACTOR-MEAN has the architecture as below:")
        if image:
            kernel_sz = kwargs.get('kernel_sz')
            n_channel = kwargs.get('n_channel')
            self.mean = ConvNet(mlp_dimList=dimList,
                                cnn_kernel_size=kernel_sz,
                                cnn_channel_numbers=n_channel,
                                mlp_act=actType,
                                mlp_output_act='Tanh',
                                img_size=img_sz,
                                verbose=verbose).to(device)
        else:
            self.mean = MLP(dimList, 
                            actType, 
                            output_activation=nn.Tanh,
                            verbose=verbose).to(device)
        # self.noise = Normal(0., noiseStd)
        self.actionSpace = actionSpace
        self.a_max = self.actionSpace.high[0]
        self.a_min = self.actionSpace.low[0]
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0
        self.noiseStd = self.scale * 0.4
        self.noiseClamp = self.noiseStd * 2
        if verbose:
            print('noise-std: {:.2f}, noise-clamp: {:.2f}'.format(
                self.noiseStd, self.noiseClamp))


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