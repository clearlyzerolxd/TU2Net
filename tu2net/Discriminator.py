import einops
import numpy
import torch
import torch.nn as nn
from torch.nn import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
from Blocks import DBlock, get_conv_layer


# from My_DGMR import DGMR


class Temporal(nn.Module):
    def __init__(self):
        super(Temporal, self).__init__()
        self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.D_1 = DBlock(input_channels=4,output_channels=48,conv_type="3d",first_relu=False)
        self.D_2 = DBlock(input_channels=48, output_channels=96, conv_type="3d", first_relu=False)
        self.drpout = nn.Dropout(0.5,inplace=True)
        self.end_d = nn.Sequential(
            DBlock(input_channels=96,output_channels=192),
            DBlock(input_channels=192, output_channels=384),
            DBlock(input_channels=384,output_channels=768),
            DBlock(input_channels=768,output_channels=768,keep_same_output=True)
        )
        self.Flatten = nn.Flatten(1)
        self.relu = torch.nn.ReLU()
      
        self.end = nn.Sequential(
            torch.nn.BatchNorm1d(768),
            spectral_norm(torch.nn.Linear(768, 1)))

    def forward(self,x):
        x = self.downsample(x)

        x = self.space2depth(x)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        x = self.D_1(x)
        x = self.D_2(x)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        x = einops.rearrange(x,"b t c w h -> (b t) c w h")


        x = self.end_d(x)

        x = torch.sum(self.relu(x),dim=[2,3])

        x = self.end(x)

        x = einops.repeat(x,"(b t) n ->b t n",t=2)
        return x







class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        self.downSample = nn.AvgPool2d(2)
        self.s2d = PixelUnshuffle(downscale_factor=2)
        self.d = nn.Sequential(
            DBlock(input_channels=4,output_channels=48),
            DBlock(input_channels=48,output_channels=96),
            DBlock(input_channels=96,output_channels=192),
            DBlock(input_channels=192,output_channels=384),
            DBlock(input_channels=384,output_channels=384,keep_same_output=True),
            DBlock(input_channels=384,output_channels=768),
        )

        self.end = nn.Sequential(
            nn.BatchNorm1d(768),
            spectral_norm(nn.Linear(768,1))
        )

        self.relu = torch.nn.ReLU()


    def forward(self,x):
        output = []
        # print(x.shape)
        for i in range(6):
            x_ = x[:,i,:,:,:]
            x_ = self.downSample(x_)
            x_ = self.s2d(x_)
            x_ = self.d(x_)


            x_ = torch.sum(self.relu(x_),dim=[2,3])

            x_ =self.end(x_)
            output.append(x_)

        output = torch.stack(output,dim=1)

        return output

class Block(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        kernelsize_h_w = 3,
        kernelsize_d=1,
        step=1,
        step_hw=1

    ):

        super().__init__()
        self.step_hw = step_hw
        self.step = step
        self.inchannel = input_channels
        self.outchannel = output_channels
        self.maxpool = nn.MaxPool3d(kernel_size=(kernelsize_d,1,1),stride=(step,2,2))
        self.conv_3x3_3D_1= spectral_norm(nn.Conv3d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1,1),stride=(1,1,1)))
        self.conv_3x3_3D_2 = spectral_norm(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=(kernelsize_d,3,3),stride=(step, step_hw, step_hw),padding=(0,1,1)))
        self.conv_3x3_3D_3 = spectral_norm(nn.Conv3d(in_channels=input_channels, out_channels=output_channels, kernel_size=(1,1,1),stride=(1, 1, 1)))
        self.re = spectral_norm(nn.Conv3d(in_channels=input_channels,out_channels=output_channels,kernel_size=(1,1,1),stride=(1,1,1)))
        self.relu = torch.nn.ReLU(True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.outchannel==self.inchannel):
            x_ = x
        else:
            x_ = x
            if(self.step_hw!=1):

                x_ = self.maxpool(x_)

            x_ = self.relu(self.re(x_))

        x = self.relu(self.conv_3x3_3D_1(x))

        x = self.relu(self.conv_3x3_3D_2(x))
        x= self.relu(self.conv_3x3_3D_3(x))

        return x+x_

class time_modify(nn.Module):
    def __init__(self):
        super(time_modify, self).__init__()
        self.s2d = PixelUnshuffle(downscale_factor=2)
        self.Block_1 = Block(input_channels=4,output_channels=48)
        self.Block_2 = Block(input_channels=48, output_channels=48)

        self.Block_3 = Block(input_channels=48, output_channels=96,step_hw=2)
        self.Block_3_1 = Block(input_channels=96, output_channels=96)
        self.Block_4 = Block(input_channels=96, output_channels=192, step_hw=2,kernelsize_d=3)
        self.Block_4_1 = Block(input_channels=192, output_channels=192)
        self.endConv = spectral_norm(nn.Conv3d(in_channels=192,out_channels=192,kernel_size=(3,4,4),stride=(1,4,4)))
        self.endConv2 = spectral_norm(nn.Conv3d(in_channels=192, out_channels=384, kernel_size=(2, 3, 3), stride=(1, 1, 1)))
        # self.Dropout =nn.Dropout(0.5)
        self.avg3d = nn.AvgPool3d((1,6,6))
        self.fc = nn.Linear(384,2)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        x = self.s2d(x)

        x = torch.permute(x, [0, 2, 1, 3, 4])

        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_2(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x= self.Block_3_1(x)
        x = self.Block_3_1(x)
        x = self.Block_3_1(x)
        x = self.Block_4(x)
        x = self.Block_4_1(x)


        x = self.relu(self.endConv(x))

        x = self.relu(self.endConv2(x))

        # x = self.Dropout(x)
        x= self.avg3d(x)

        x = torch.squeeze(x)
        x = self.fc(x)

        return x

# lable_all = torch.tensor([1,0])
# x = torch.nn.functional.one_hot(lable_all)# loss = nn.CrossEntropyLoss()
# print(x[0].shape)
# x = torch.unsqueeze(x[0],dim=0)
# print(x)
# lable_real = einops.repeat(x, "w  h  -> (repeat w)  h  ", repeat=8)# print(torch.cuda.memory_allocated() / 1024 / 1024)
# print(lable_real)
# x = torch.rand(size=(8,6,1,256,256)).to("cuda:0")
# # x = torch.permute(x,[0,2,1,3,4])
# net = time_modify().to("cuda:0")
# x = net(x)
# print(x.shape)
# loss(x,torch.ones_like(x))
# print(x.shape)
# print(torch.cuda.memory_allocated() / 1024 / 1024)
# x = torch.randn(size=(4,18,1,256,256))
# net = Temporal()
# y = net(x)
# print(y.shape)

# net1 = nn.Sequential(
#     nn.Linear(30, 40),
#     nn.ReLU(),
#     nn.Linear(40, 50),
#     nn.ReLU(),
#
#     nn.Linear(50, 10)
# )
# x = torch.rand(size=(2,30))*100000
# y = torch.ones(size=(2,30))
# print(torch.sum(net1(x)))
# print(torch.sum(net1(y)))
#
# x = torch.rand(size=(8,6,1,256,256))
# Dis = Spatial()
# x = Dis(x)
# print(x)
# print(torch.mean(x))
# class TemporalDiscriminator(torch.nn.Module):
#     def __init__(
#         self
#     ):
#
#         super().__init__()
#
#
#         self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.space2depth = PixelUnshuffle(downscale_factor=2)
#         internal_chn = 48
#         self.d1 = DBlock(
#             input_channels=4,
#             output_channels=48,
#             conv_type="3d",
#             first_relu=False,
#         )
#         self.d2 = DBlock(
#             input_channels=48,
#             output_channels=96,
#             conv_type="3d",
#         )
#         # self.intermediate_dblocks = torch.nn.ModuleList()
#         # for _ in range(6):
#         #
#         #     self.intermediate_dblocks.append(
#         #         DBlock(
#         #             input_channels=internal_chn * input_channels,
#         #             output_channels=2 * internal_chn * input_channels,
#         #             conv_type=conv_type,
#         #         )
#         #     )
#         #
#         # self.d_last = DBlock(
#         #     input_channels=2 * internal_chn * input_channels,
#         #     output_channels=2 * internal_chn * input_channels,
#         #     keep_same_output=True,
#         #     conv_type=conv_type,
#         # )
#         self.d = nn.Sequential(
#             DBlock(input_channels=4,output_channels=48),
#             DBlock(input_channels=48,output_channels=96),
#             DBlock(input_channels=96,output_channels=192),
#             DBlock(input_channels=192,output_channels=384),
#             DBlock(input_channels=384,output_channels=384,keep_same_output=True),
#             DBlock(input_channels=384,output_channels=768),
#         )
#         self.fc = spectral_norm(torch.nn.Linear(768, 1))
#         self.relu = torch.nn.ReLU()
#         self.bn = torch.nn.BatchNorm1d(768)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.downsample(x)
#
#         x = self.space2depth(x)
#         # Have to move time and channels
#         x = torch.permute(x, dims=(0, 2, 1, 3, 4))
#         # 2 residual 3D blocks to halve resolution if image, double number of channels and reduce
#         # number of time steps
#         x = self.d1(x)
#         x = self.d2(x)
#         # Convert back to T x C x H x W
#         x = torch.permute(x, dims=(0, 2, 1, 3, 4))
#         # Per Timestep part now, same as spatial discriminator
#         representations = []
#         for idx in range(x.size(1)):
#             # Intermediate DBlocks
#             # Three residual D Blocks to halve the resolution of the image and double
#             # the number of channels.
#             rep = x[:, idx, :, :, :]
#             # for d in self.intermediate_dblocks:
#             #     rep = d(rep)
#             rep = self.d(rep)
#             # One more D Block without downsampling or increase number of channels
#             # rep = self.d_last(rep)
#
#             rep = torch.sum(F.relu(rep), dim=[2, 3])
#             rep = self.bn(rep)
#             rep = self.fc(rep)
#
#             representations.append(rep)
#         # The representations are summed together before the ReLU
#         x = torch.stack(representations, dim=1)
#         # Should be [Batch, N, 1]
#         x = torch.sum(x, keepdim=True, dim=1)
#         return x
#
# x = torch.rand(size=(4,6,1,256,256))
# net = Spatial()
# x  = net(x)
# print(x.shape)
