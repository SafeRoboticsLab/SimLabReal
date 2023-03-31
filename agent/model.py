# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for building blocks for actors and critics.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agent.neural_network import MLP, ConvNet


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder(torch.nn.Module):
    """Conv layers shared by actor and critic in SAC."""

    def __init__(
        self,
        input_n_channel,
        img_sz,
        kernel_sz,
        stride,
        n_channel,
        use_sm=True,
        use_spec=False,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose: bool = True,
    ):
        super().__init__()
        if verbose:
            print(
                "The neural network for encoder has the architecture as below:"
            )
        self.conv = ConvNet(
            input_n_channel=input_n_channel, cnn_kernel_size=kernel_sz,
            cnn_stride=stride, output_n_channel=n_channel, img_size=img_sz,
            use_sm=use_sm, use_spec=use_spec, use_bn=use_bn,
            use_residual=use_residual, verbose=verbose
        ).to(device)

    def forward(self, image, detach=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        out = self.conv(image)
        if detach:
            out = out.detach()
        return out

    def copy_conv_weights_from(self, source):
        """
        Tie convolutional layers - assume actor and critic have same conv
        structure.
        """
        for source_module, module in zip(
            source.conv.moduleList, self.conv.moduleList
        ):
            for source_layer, layer in zip(
                source_module.children(), module.children()
            ):  # children() works for both Sequential and nn.Module
                if isinstance(layer, nn.Conv2d):
                    tie_weights(src=source_layer, trg=layer)

    def get_output_dim(self):
        return self.conv.get_output_dim()


class SACPiNetwork(torch.nn.Module):

    def __init__(
        self,
        input_n_channel,
        mlp_dim,
        action_dim,
        action_mag,
        activation_type,  # for MLP; ReLU default for conv
        img_sz,
        kernel_sz,
        stride,
        n_channel,
        latent_dim=0,
        latent_dim_cnn=0,
        append_dim=0,
        use_sm=True,
        use_ln=True,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose: bool = True,
    ):
        super().__init__()
        self.latent_dim_cnn = latent_dim_cnn
        self.img_sz = img_sz
        if np.isscalar(img_sz):
            self.img_sz = [img_sz, img_sz]

        # Conv layers shared with critic
        self.encoder = Encoder(
            input_n_channel=input_n_channel + latent_dim_cnn, img_sz=img_sz,
            kernel_sz=kernel_sz, stride=stride, n_channel=n_channel,
            use_sm=use_sm, use_spec=False, use_bn=use_bn,
            use_residual=use_residual, device=device, verbose=False
        )
        if use_sm:
            dim_conv_out = n_channel[-1] * 2  # assume spatial softmax
        else:
            dim_conv_out = self.encoder.get_output_dim()

        # Linear layers
        # Latent dimension appended to the input of MLP is
        # (latent_dim-latent_dim_cnn).
        mlp_dim = [dim_conv_out + append_dim +
                   (latent_dim-latent_dim_cnn)] + mlp_dim + [action_dim]
        self.mlp = GaussianPolicy(
            mlp_dim, action_mag, activation_type, use_ln, device, verbose
        )

    def forward(
        self,
        image,  # NCHW
        append,  # N x append_dim
        latent=None,  # N x z_dim
        detach_encoder=False,
    ):
        """
        Assume all arguments have the same number of leading dims (L and N),
        and returns the same number of leading dims. init_rnn_state is always
        L=1.
        """
        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(next(self.parameters()).device)
            np_input = True
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape

        # Append latent to image input
        if self.latent_dim_cnn > 0:
            latent_cnn = latent[:, :self.latent_dim_cnn]
            latent_cnn = latent_cnn.unsqueeze(-1).unsqueeze(
                -1
            )  # make H, W channels
            latent_cnn = latent_cnn.repeat(1, 1, H, W)
            image = torch.cat((image, latent_cnn), dim=-3)  # dim=C

        # Forward thru conv
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Append, latent
        conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            latent_mlp = latent[:, self.latent_dim_cnn:]
            conv_out = torch.cat((conv_out, latent_mlp), dim=-1)

        # MLP
        output = self.mlp(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            output = output.squeeze(0)

        # Convert back to np
        if np_input:
            output = output.detach().cpu().numpy()
        return output

    def sample(self, image, append, latent=None, detach_encoder=False):
        """
        Assume all arguments have the same number of leading dims (N),
        and returns the same number of leading dims.
        """
        # Convert to torch
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(next(self.parameters()).device)
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape

        # Append latent to image channels
        if self.latent_dim_cnn > 0:
            latent_cnn = latent[:, :self.latent_dim_cnn]
            latent_cnn = latent_cnn.unsqueeze(-1).unsqueeze(
                -1
            )  # make H, W channels
            latent_cnn = latent_cnn.repeat(1, 1, H, W)
            image = torch.cat((image, latent_cnn), dim=-3)  # dim=C

        # Get CNN output
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Append, latent
        conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            latent_mlp = latent[:, self.latent_dim_cnn:]
            conv_out = torch.cat((conv_out, latent_mlp), dim=-1)

        # MLP
        action, log_prob = self.mlp.sample(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return (action, log_prob)

    def sample_and_MI(
        self, image, append, latent, latent_samples, detach_encoder=False
    ):
        """
        Samples the action and get the likelihood and the marginal given that
        action.

        Args:
            image ([type]): [description]
            append ([type]): [description]
            latent ([type]): [description]
            latent_samples ([type]): [description]
            detach_encoder (bool, optional): [description]. Defaults to False.

        Raises:
            NotImplementedError: [description]
        """
        # Convert to torch
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(next(self.parameters()).device)
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape

        # Append latent to image channels
        if self.latent_dim_cnn > 0:
            latent_cnn = latent[:, :self.latent_dim_cnn]
            latent_cnn = latent_cnn.unsqueeze(-1).unsqueeze(
                -1
            )  # make H, W channels
            latent_cnn = latent_cnn.repeat(1, 1, H, W)
            image_reg = torch.cat((image, latent_cnn), dim=-3)  # dim=C

        # Get CNN output (already flattened)
        conv_out = self.encoder.forward(image_reg, detach=detach_encoder)

        # Add append
        conv_out = torch.cat((conv_out, append), dim=-1)

        # Add latent
        if self.latent_dim_cnn > 0:
            latent_mlp = latent[:, self.latent_dim_cnn:]
            conv_out = torch.cat((conv_out, latent_mlp), dim=-1)
        else:
            conv_out = torch.cat((conv_out, latent), dim=-1)

        # Forward thru mlp
        action, log_prob, rv = self.mlp.sample(conv_out, get_x=True)

        # repeat latents
        num_z_per_image = int(latent_samples.shape[0] / image.shape[0])

        # repeat image, append, rv
        conv_input = torch.repeat_interleave(image, num_z_per_image, dim=0)
        append_repeat = torch.repeat_interleave(append, num_z_per_image, dim=0)
        rv_repeat = torch.repeat_interleave(rv, num_z_per_image, dim=0)

        # get pdf
        if self.latent_dim_cnn > 0:
            latent_samples_cnn = latent_samples[:, :self.latent_dim_cnn]
            latent_samples_cnn = latent_samples_cnn.unsqueeze(-1).unsqueeze(
                -1
            ).repeat(1, 1, H, W)
            conv_input = torch.cat((conv_input, latent_samples_cnn),
                                   dim=-3)  # dim=C
        conv_out_samp = self.encoder.forward(conv_input, detach=detach_encoder)
        conv_out_samp = torch.cat((conv_out_samp, append_repeat), dim=-1)

        if self.latent_dim_cnn > 0:
            latent_samples_mlp = latent_samples[:, self.latent_dim_cnn:]
            conv_out_samp = torch.cat((conv_out_samp, latent_samples_mlp),
                                      dim=-1)
        else:
            conv_out_samp = torch.cat((conv_out_samp, latent_samples), dim=-1)
        log_prob_samp = self.mlp.get_pdf(conv_out_samp, rv_repeat)

        # Get marginal
        prob_sample_avg = log_prob_samp.view(N, num_z_per_image
                                            ).exp().mean(dim=1, keepdim=True)
        log_marginal = (prob_sample_avg + 1e-8).log()

        # Restore dimension
        for _ in range(num_extra_dim):
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return (action, log_prob, log_marginal)

    def get_log_marginal(
        self, image, append, action, latent_samples, detach_encoder=False
    ):
        # Convert to torch
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(next(self.parameters()).device)
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape

        # get the random variable
        with torch.no_grad():
            y = (action - self.mlp.bias) / self.mlp.scale
            rv = torch.atanh(y)

        # repeat latents
        num_z_per_image = int(latent_samples.shape[0] / image.shape[0])

        # repeat image, append, rv
        conv_input = torch.repeat_interleave(image, num_z_per_image, dim=0)
        append_repeat = torch.repeat_interleave(append, num_z_per_image, dim=0)
        rv_repeat = torch.repeat_interleave(rv, num_z_per_image, dim=0)

        # get pdf
        if self.latent_dim_cnn > 0:
            latent_samples_cnn = latent_samples[:, :self.latent_dim_cnn]
            latent_samples_cnn = latent_samples_cnn.unsqueeze(-1).unsqueeze(
                -1
            ).repeat(1, 1, H, W)
            conv_input = torch.cat((conv_input, latent_samples_cnn),
                                   dim=-3)  # dim=C

        conv_out_samp = self.encoder.forward(conv_input, detach=detach_encoder)
        conv_out_samp = torch.cat((conv_out_samp, append_repeat), dim=-1)

        if self.latent_dim_cnn > 0:
            latent_samples_mlp = latent_samples[:, self.latent_dim_cnn:]
            conv_out_samp = torch.cat((conv_out_samp, latent_samples_mlp),
                                      dim=-1)
        else:
            conv_out_samp = torch.cat((conv_out_samp, latent_samples), dim=-1)

        log_prob_samp = self.mlp.get_pdf(conv_out_samp, rv_repeat)

        # Get marginal
        prob_sample_avg = log_prob_samp.view(N, num_z_per_image
                                            ).exp().mean(dim=1, keepdim=True)
        log_marginal = (prob_sample_avg + 1e-8).log()

        return log_marginal


class SACTwinnedQNetwork(torch.nn.Module):

    def __init__(
        self,
        input_n_channel,
        mlp_dim,
        action_dim,
        activation_type,  # for MLP; ReLU default for conv
        img_sz,
        kernel_sz,
        stride,
        n_channel,
        latent_dim=0,
        latent_dim_cnn=0,
        append_dim=0,
        use_sm=True,
        use_ln=True,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose: bool = True,
    ):
        super().__init__()
        self.img_sz = img_sz
        if np.isscalar(img_sz):
            self.img_sz = [img_sz, img_sz]
        self.latent_dim_cnn = latent_dim_cnn

        # Conv layers shared with critic
        self.encoder = Encoder(
            input_n_channel=input_n_channel + latent_dim_cnn, img_sz=img_sz,
            kernel_sz=kernel_sz, stride=stride, n_channel=n_channel,
            use_sm=use_sm, use_spec=False, use_bn=use_bn,
            use_residual=use_residual, device=device, verbose=False
        )
        if use_sm:
            dim_conv_out = n_channel[-1] * 2  # assume spatial softmax
        else:
            dim_conv_out = self.encoder.get_output_dim()

        # Latent dimension appended to the input of MLP is
        # (latent_dim-latent_dim_cnn).
        mlp_dim = [
            dim_conv_out +
            (latent_dim-latent_dim_cnn) + append_dim + action_dim
        ] + mlp_dim + [1]
        self.Q1 = MLP(
            mlp_dim, activation_type, out_activation_type='Identity',
            use_ln=use_ln, verbose=False
        ).to(device)
        self.Q2 = copy.deepcopy(self.Q1)
        if verbose:
            print("The MLP for critic has the architecture as below:")
            print(self.Q1.moduleList)

    def forward(
        self, image, actions, append, latent=None, detach_encoder=False
    ):
        """
        Assume all arguments have the same number of leading dims (L and N),
        and returns the same number of leading dims.
        """

        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(next(self.parameters()).device)
            np_input = True
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            actions = actions.unsqueeze(0)
            append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape

        # Append latent to image channels
        if self.latent_dim_cnn > 0:
            latent_cnn = latent[:, :self.latent_dim_cnn]
            latent_cnn = latent_cnn.unsqueeze(-1).unsqueeze(
                -1
            )  # make H, W channels
            latent_cnn = latent_cnn.repeat(1, 1, H, W)
            image = torch.cat((image, latent_cnn), dim=-3)  # dim=C

        # Get CNN output
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Append, latent
        conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            latent_mlp = latent[:, self.latent_dim_cnn:]
            conv_out = torch.cat((conv_out, latent_mlp), dim=-1)

        # Append action to mlp
        conv_out = torch.cat((conv_out, actions), dim=-1)

        # MLP
        q1 = self.Q1(conv_out)
        q2 = self.Q2(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            q1 = q1.squeeze(0)
            q2 = q2.squeeze(0)

        # Convert back to np
        if np_input:
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()
        return q1, q2


#== Policy (Actor) Model ==
class GaussianPolicy(nn.Module):

    def __init__(
        self, dimList, action_mag, activation_type='ReLU', use_ln=True,
        device='cpu', verbose: bool = True
    ):
        super(GaussianPolicy, self).__init__()
        self.mean = MLP(
            dimList, activation_type, out_activation_type='Identity',
            use_ln=use_ln, verbose=False
        ).to(device)
        self.log_std = MLP(
            dimList, activation_type, out_activation_type='Identity',
            use_ln=use_ln, verbose=False
        ).to(device)
        if verbose:
            print("The MLP for MEAN has the architecture as below:")
            print(self.mean.moduleList)
            # print("The MLP for LOG_STD has the architecture as below:")
            # print(self.log_std.moduleList)

        self.a_max = action_mag
        self.a_min = -action_mag
        self.scale = (self.a_max - self.a_min) / 2.0  # basically the mag
        self.bias = (self.a_max + self.a_min) / 2.0
        self.LOG_STD_MAX = 1
        self.LOG_STD_MIN = -10
        self.eps = 1e-8

    def forward(self, state):  # mean only
        state_tensor = state.to(next(self.parameters()).device)
        mean = self.mean(state_tensor)
        mean = torch.tanh(mean)
        return mean * self.scale + self.bias

    def sample(self, state, get_x=False):
        state_tensor = state.to(next(self.parameters()).device)
        mean = self.mean(state_tensor)
        log_std = self.log_std(state_tensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Sample
        normal_rv = Normal(mean, std)
        x = normal_rv.rsample(
        )  # reparameterization trick (mean + std * N(0,1))
        log_prob = normal_rv.log_prob(x)

        # Get action
        y = torch.tanh(x)  # constrain the output to be within [-1, 1]
        action = y * self.scale + self.bias
        # Get the correct probability: x -> a, a = c * y + b, y = tanh x
        # followed by: p(a) = p(x) x |det(da/dx)|^-1
        # log p(a) = log p(x) - log |det(da/dx)|
        # log |det(da/dx)| = sum log (d a_i / d x_i)
        # d a_i / d x_i = c * ( 1 - y_i^2 )
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(log_prob.dim() - 1, keepdim=True)
        else:
            log_prob = log_prob.sum()
        # mean = torch.tanh(mean) * self.scale + self.bias
        if get_x:
            return action, log_prob, x
        return action, log_prob

    def get_pdf(self, state, x):
        # either state is a single vector or action is a single vector
        state_tensor = state.to(next(self.parameters()).device)
        mean = self.mean(state_tensor)
        log_std = self.log_std(state_tensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        normal_rv = Normal(mean, std)
        log_prob = normal_rv.log_prob(x)

        y = torch.tanh(x)
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(log_prob.dim() - 1, keepdim=True)
        else:
            log_prob = log_prob.sum()
        return log_prob


class Discriminator(torch.nn.Module):

    def __init__(
        self,
        input_n_channel,  # usually 2 frames
        mlp_dim,
        latent_dim,
        img_sz,
        kernel_sz,
        stride,
        n_channel,
        append_dim=0,
        use_spec=True,
        use_sm=False,
        use_ln=False,
        device='cpu',
        verbose: bool = True,
    ):
        super().__init__()
        self.img_sz = img_sz
        if np.isscalar(img_sz):
            self.img_sz = [img_sz, img_sz]

        # Conv layers shared with critic
        if verbose:
            print("The encoder has the architecture as below:")
        self.encoder = Encoder(
            input_n_channel=input_n_channel, img_sz=img_sz,
            kernel_sz=kernel_sz, stride=stride, n_channel=n_channel,
            use_sm=use_sm, use_spec=use_spec, use_bn=False, use_residual=False,
            device=device, verbose=False
        )
        if use_sm:
            dim_conv_out = n_channel[-1] * 2  # assume spatial softmax
        else:
            dim_conv_out = self.encoder.get_output_dim()
        mlp_dim = [dim_conv_out+append_dim] + mlp_dim + [
            latent_dim
        ]  # stack features from consecutive frames

        # Linear layers
        if verbose:
            print("The MLP has the architecture as below:")
        self.mean = MLP(
            mlp_dim, activation_type='ReLU', use_ln=use_ln, use_spec=use_spec,
            verbose=verbose
        ).to(device)  # use spectral norm

    def forward(self, image, append, latent, detach_encoder=False):
        """
        Get log probability directly
        """
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(
                next(self.parameters()).device
            )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Make batch
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            B = 1
        else:
            B = image.shape[0]

        # Forward pass
        conv_out = self.encoder.forward(image,
                                        detach=detach_encoder).view(B, -1)
        conv_out = torch.cat((conv_out, append), dim=1)
        mean = self.mean(conv_out)

        # Learned variance
        # log_std = self.log_std(conv_out)
        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        # std = torch.exp(log_std)
        # normal_rv = Normal(mean, std)
        # log_prob = normal_rv.log_prob(latent)
        # print(mean[-1], std[-1], latent[-1], log_prob[-1])

        # Fixed variance, sigma = 1.0
        log_prob = -(mean - latent)**2
        return log_prob
