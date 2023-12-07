import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from TU2Net import Generator_full
from Discriminator import Spatial,Temporal







def train():
    assert torch.cuda.is_available(), "CUDA is not available. Training on CPU is not supported."
    print("##### Build The train #######")
    device = "cuda"
    
    Generate_net = Generator_full(frames=6).to(device)
    sum_Generate_net = sum(p.numel() for p in Generate_net.parameters())* 4 / (1024 ** 2)
    print("Generate_net total number of parameters is {}MB".format(round(sum_Generate_net,2 )))
    
    Spatial_dis = Spatial().to(device)
    Temporal_dis = Temporal().to(device)
    Spatial_dis_sum = sum(p.numel() for p in Spatial_dis.parameters())* 4 / (1024 ** 2)
    
    Temporal_dis_sum = sum(p.numel() for p in Temporal_dis.parameters())* 4 / (1024 ** 2)
    
    
    
    Generate_net_optim = optim.Adam(Generate_net.parameters(),lr=2e-4,betas=(0.0, 0.999))
    
    Spatial_dis_optim = optim.Adam(Spatial_dis.parameters(),lr=2e-5,betas=(0.0,0.999))
    
    Temporal_dis_optim = optim.Adam(Temporal_dis.parameters(),lr=2e-5,betas=(0.0,0.999))
    
    
    
    
    print("Spatial_dis_sum total number of parameters is {}MB \nTemporal_dis_sum total number of parameters is {}MB".format(round(Temporal_dis_sum,2),round(Spatial_dis_sum,2)))
    
    
    
    
    
    
    
    
train()