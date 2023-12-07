import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from TU2Net import Generator_full








def train():
    assert torch.cuda.is_available(), "CUDA is not available. Training on CPU is not supported."
    print("##### Build The train #######")
    device = "cuda"
    
    net = Generator_full()
    
train()