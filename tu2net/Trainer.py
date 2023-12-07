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
    sum_Generate_net = sum(p.numel() for p in Generate_net.parameters())
    print("Generate_net total number of parameters is {}MB".format(round(sum_Generate_net* 4 / (1024 ** 2),2 )))
    
    Spatial_dis = Spatial()
    Temporal_dis = Temporal()
    Spatial_dis_sum = sum(p.numel() for p in Spatial_dis.parameters())
    
    Temporal_dis_sum = sum(p.numel() for p in Temporal_dis.parameters())
    
    
    print()
    
    
    
    
    
    
    
    
train()