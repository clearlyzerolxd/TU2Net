from typing import Union, Type
from torch.distributions import normal
import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv3d, PixelUnshuffle
import math
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
import torch
from Attention import AttentionLayer
# from Attention import AttentionLayer


def get_conv_layer(conv_type: str = "standard") -> Type[Union[Conv2d, Conv3d]]:
    if conv_type == "standard":
        conv_layer = torch.nn.Conv2d
    elif conv_type == "3d":
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer

class ConvGRU(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size=3, sn_eps=0.0001):
        super(ConvGRU, self).__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps
        self.output_channels=output_channels

        self.read_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.output_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        # self.att = AttentionLayer(input_channels=64,output_channels=64)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(True)
    def forward(self,x,prev_state):
        # x = self.att(x)
        # prev_state = self.att(prev_state)
        xh = torch.cat([x, prev_state], dim=1)

        read_gate =  self.sig(self.read_gate_conv(xh))
        update_gate = self.sig(self.update_gate_conv(xh))

        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        c = self.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out

        return out, new_state


class DBlock(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        first_relu: bool = True,
        keep_same_output: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        )
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )
        self.relu = torch.nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:


        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x
        return x
class My_GRU(nn.Module):
    def __init__(self, input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps=0.0001,):
        super(My_GRU, self).__init__()
        self.GRU = ConvGRU(input_channels,output_channels,kernel_size,sn_eps)
        self.output_channels = output_channels

    def forward(self,x,pre_state):
        out_cat = []
        sum_ = 0
        pre_pre_pre_state = 0
        pre_pre_state = 0
        for i in range(len(x)):
            pre_state_ = pre_state
            # print("->>",x[i].shape,pre_state.shape)
            output, pre_state = self.GRU(x[i], pre_state)

            # if (i ==0):
            #     pre_pre_pre_state = pre_state
            # elif(i==1):
            #     pre_pre_state = pre_state
            # else:
            #     pre_state +=pre_pre_pre_state+pre_state
            #     pre_pre_pre_state = pre_pre_state
            #     pre_pre_state = pre_state

            # print("cat_",pre_state_.shape)

            output = torch.cat([output,pre_state_],dim=1)
            # print("cat_later",output.shape)
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            # print("cat_later_interpolate", output.shape)
            out_cat.append(output)
        outputs = torch.stack(out_cat, dim=0)

        return outputs

class GBlock(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):

        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            eps=spectral_normalized_eps,
        )
        # Upsample 2D conv
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally spectrally normalized 1x1 convolution
        if x.shape[1] != self.output_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        x = x2 + sc
        return x


class UpsampleGBlock(torch.nn.Module):
    """Residual generator block with upsampling"""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor):

        sc = self.upsample(x)
        sc = self.conv_1x1(sc)
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + sc
        return x

# class LBlock(torch.nn.Module):
#     def __init__(
#         self,
#         input_channels: int = 12,
#         output_channels: int = 12,
#         kernel_size: int = 3,
#         conv_type: str = "standard",
#     ):
#         super().__init__()
#
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         conv2d = get_conv_layer(conv_type)
#         self.conv_1x1 = conv2d(
#             in_channels=input_channels,
#             out_channels=output_channels - input_channels,
#             kernel_size=1,
#         )
#
#         self.first_conv_3x3 = conv2d(
#             input_channels,
#             out_channels=output_channels,
#             kernel_size=kernel_size,
#             padding=1,
#             stride=1,
#         )
#         self.relu = torch.nn.ReLU()
#         self.last_conv_3x3 = conv2d(
#             in_channels=output_channels,
#             out_channels=output_channels,
#             kernel_size=kernel_size,
#             padding=1,
#             stride=1,
#         )
#
#     def forward(self, x) -> torch.Tensor:
#
#         if self.input_channels < self.output_channels:
#             sc = self.conv_1x1(x)
#             sc = torch.cat([x, sc], dim=1)
#         else:
#             sc = x
#
#         x2 = self.relu(x)
#         x2 = self.first_conv_3x3(x2)
#         x2 = self.relu(x2)
#         x2 = self.last_conv_3x3(x2)
#         return x2 + sc
# class LatentConditioningStack(torch.nn.Module):
#     def __init__(
#         self,
#         shape: (int, int, int) = (8, 8, 8),
#         output_channels: int = 768,
#         use_attention: bool = True,
#
#     ):
#         super().__init__()
#
#
#         self.shape = shape
#         self.use_attention = use_attention
#         self.distribution = normal.Normal(loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0]))
#
#         self.conv_3x3 = spectral_norm(
#             torch.nn.Conv2d(
#                 in_channels=shape[0], out_channels=shape[0], kernel_size=(3, 3), padding=1
#             )
#         )
#         self.l_block1 = LBlock(input_channels=shape[0], output_channels=output_channels // 32)
#         self.l_block2 = LBlock(
#             input_channels=output_channels // 32, output_channels=output_channels // 16
#         )
#         self.l_block3 = LBlock(
#             input_channels=output_channels // 16, output_channels=output_channels // 4
#         )
#         if self.use_attention:
#             self.att_block = AttentionLayer(
#                 input_channels=output_channels // 4, output_channels=output_channels // 4
#             )
#         self.l_block4 = LBlock(input_channels=output_channels // 4, output_channels=output_channels)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#
#         z = self.distribution.sample(self.shape)
#
#         z = torch.permute(z, (3, 0, 1, 2)).type_as(x)
#
#         z = self.conv_3x3(z)
#
#         z = self.l_block1(z)
#         z = self.l_block2(z)
#         z = self.l_block3(z)
#
#         z = self.att_block(z)
#
#         z = self.l_block4(z)
#         return z
class LBlock(torch.nn.Module):


    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        kernel_size: int = 3,
        conv_type: str = "standard",
    ):
        """
        L-Block for increasing the number of channels in the input
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        # Output size should be channel_out - channel_in
        self.input_channels = input_channels
        self.output_channels = output_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels - input_channels,
            kernel_size=1,
        )

        self.first_conv_3x3 = conv2d(
            input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        self.relu = torch.nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )

    def forward(self, x) -> torch.Tensor:

        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = torch.cat([x, sc], dim=1)
        else:
            sc = x

        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        return x2 + sc
class LatentConditioningStack(torch.nn.Module):
    def __init__(
        self,
        shape: (int, int, int) = (8, 8, 8),
        output_channels: int = 768,
        use_attention: bool = True,

    ):
        super().__init__()


        self.shape = shape
        self.use_attention = use_attention
        self.distribution = normal.Normal(loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0]))

        self.conv_3x3 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=shape[0], out_channels=shape[0], kernel_size=(3, 3), padding=1
            )
        )
        self.l_block1 = LBlock(input_channels=shape[0], output_channels=output_channels // 32)
        self.l_block2 = LBlock(
            input_channels=output_channels // 32, output_channels=output_channels // 16
        )
        self.l_block3 = LBlock(
            input_channels=output_channels // 16, output_channels=output_channels // 4
        )
        if self.use_attention:
            self.att_block = AttentionLayer(
                input_channels=output_channels // 4, output_channels=output_channels // 4
            )
        self.l_block4 = LBlock(input_channels=output_channels // 4, output_channels=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.distribution.sample(self.shape)

        z = torch.permute(z, (3, 0, 1, 2)).type_as(x)


        z = self.conv_3x3(z)

        z = self.l_block1(z)
        z = self.l_block2(z)
        z = self.l_block3(z)

        z = self.att_block(z)

        z = self.l_block4(z)
        return z


class attblock(nn.Module):
    def __init__(self):
        super(attblock, self).__init__()
        self.att = AttentionLayer(
            input_channels=512 , output_channels=512
        )
    def forward(self,x):
        x = self.att(x)
        return (x)

# x  = torch.rand(size=(8,64,4,4))
# net = lblock()
# print(net(x).shape)