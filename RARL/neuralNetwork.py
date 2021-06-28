# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sin(nn.Module):
    """
    Sin: Wraps element-wise `sin` activation as a nn.Module.

    Shape:
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Output: `(N, *)`, same shape as the input

    Examples:
        >>> m = Sin()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return torch.sin(input) # simply apply already implemented sin


class MLP(nn.Module):
    """
    model: Constructs a fully-connected neural network with flexible depth, width
        and activation function choices.
    """
    def __init__(self, dimList, actType='Tanh', output_activation=nn.Identity, verbose=False):
        """
        __init__: Initalizes.

        Args:
            dimList (int List): the dimension of each layer.
            actType (str, optional): the type of activation function. Defaults to 'Tanh'.
                Currently supports 'Sin', 'Tanh' and 'ReLU'.
            verbose (bool, optional): print info or not. Defaults to False.
        """
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not added to
        # computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        numLayer = len(dimList)-1
        for idx in range(numLayer):
            i_dim = dimList[idx]
            o_dim = dimList[idx+1]

            self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
            if idx == numLayer-1: # final linear layer, no act.
                self.moduleList.append(output_activation())
            else:
                if actType == 'Sin':
                    self.moduleList.append(Sin())
                elif actType == 'Tanh':
                    self.moduleList.append(nn.Tanh())
                elif actType == 'ReLU':
                    self.moduleList.append(nn.ReLU())
                else:
                    raise ValueError('Activation type ({:s}) is not included!'.format(actType))
                # self.moduleList.append(nn.Dropout(p=.5))
        if verbose:
            print(self.moduleList)

        # Initalizes the weight
        # self._initialize_weights()


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.1)
    #             m.bias.data.zero_()


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)


    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        N = feature.shape[0]

        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(N, self.channel, self.height*self.width)

        softmax_attention = F.softmax(feature, dim=-1)

        # Sum over all pixels
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=2, keepdim=False)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=2, keepdim=False)
        expected_xy = torch.cat([expected_x, expected_y], 1)

        return expected_xy


class ConvNet(nn.Module):
    def __init__(   self,
                    input_channel_number=1, # not counting z_conv
                    mlp_append_dim=0,       # not counting z_mlp
                    cnn_kernel_size=[5, 3],
                    cnn_channel_numbers=[16, 32],
                    mlp_dimList=[128, 128, 1],
                    mlp_act='Tanh',
                    mlp_output_act='Tanh',
                    # num_mlp_output=5,
                    # out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
                    z_conv_dim=0,
                    z_mlp_dim=0,
                    img_size=128,
                    ):

        super(ConvNet, self).__init__()

        self.mlp_append_dim = mlp_append_dim
        self.z_conv_dim = z_conv_dim
        self.z_mlp_dim = z_mlp_dim

        # Use ModuleList to store 2 conv layers, 1 spatial softmax and [N] MLP layers
        self.moduleList = nn.ModuleList()

        #= CNN: W' = (W - kernel_size + 2*padding) / stride + 1
        kernel_size = cnn_kernel_size[0]
        padding = int( (kernel_size-1) / 2)
        self.conv_1 = nn.Sequential(  # Nx1x128x128
            nn.Conv2d(  in_channels=input_channel_number + z_conv_dim,
                        out_channels=cnn_channel_numbers[0],
                        kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            )    # Nx16x128x128

        kernel_size = cnn_kernel_size[1]
        padding = int( (kernel_size-1) / 2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(  in_channels=cnn_channel_numbers[0],
                        out_channels=cnn_channel_numbers[1],
                        kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            )    # Nx32x128x128

        #= Spatial softmax, output 64 (32 features x 2d pos)
        self.sm = SpatialSoftmax(   height=img_size,
                                    width=img_size,
                                    channel=cnn_channel_numbers[1])

        #= MLP
        cnn_output_dim = int(cnn_channel_numbers[1] * 2)
        self.linear_1 = nn.Sequential(
            nn.Linear(  cnn_output_dim+mlp_append_dim+z_mlp_dim,
                        mlp_dimList[0],
                        bias=True),
            nn.ReLU(),
            )

        self.linear_2 = nn.Sequential(
            nn.Linear(  mlp_dimList[0],
                        mlp_dimList[1],
                        bias=True),
            nn.ReLU(),
            )

        # Output action
        self.linear_out = nn.Linear(mlp_dimList[1],
                                    mlp_dimList[2],
                                    bias=True)


    def forward(self, img, zs=None, mlp_append=None):

        if img.dim() == 3:
            img = img.unsqueeze(1)  # Nx1xHxW
        N, _, H, W = img.shape

        # Attach latent to image
        if self.z_conv_dim > 0:
            zs_conv = zs[:,:self.z_conv_dim].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # repeat for all pixels, Nx(z_conv_dim)x200x200
            img = torch.cat((img, zs_conv), dim=1)  # along channel

        # CNN
        x = self.conv_1(img)
        x = self.conv_2(x)

        # Spatial softmax
        x = self.sm(x)

        # MLP, add latent as concat
        if self.z_mlp_dim > 0:
            x = torch.cat((x, zs[:,self.z_conv_dim:]), dim=1)
        if mlp_append is not None:
            x = torch.cat((x, mlp_append), dim=1)

        x = self.linear_1(x)
        x = self.linear_2(x)
        out = self.linear_out(x)

        return out