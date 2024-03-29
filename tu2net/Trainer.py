import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from TU2Net import Generator_full
from Discriminator import Spatial,Temporal
from torch.utils.data import Dataset,DataLoader
import yaml
from losses import Generator_loss_skillful,DiscriminatorLoss_hinge
import os
import numpy
from lr_scheduler import LambdaLinearScheduler
import torch.optim.lr_scheduler as lr_scheduler
from utils import rainprint
class MyDataset(Dataset):
    """rainfall data """
    def __init__(self,data_path,Normalized =False):
        super(MyDataset, self).__init__()
        self.path =os.listdir(data_path)
        self.root_path = data_path
        self.Normalized = Normalized
        print("There are {} sets of data in total".format(len(self.path)))
    def __getitem__(self, index):
        x = numpy.load(os.path.join(self.root_path,self.path[index]))
        # print(x.shape)
        
        x = torch.from_numpy(x)
        if self.Normalized:
            x = torch.clip(x,0,22.0)
            x = x/22.0
        return x[:4],x[4:10]

    def __len__(self):
        return len(self.path)




def train():
    assert torch.cuda.is_available(), "CUDA is not available. Training on CPU is not supported."
    print("##### Build The train #######")
    write = SummaryWriter()
    
    device = "cuda"
    
    Generate_net = Generator_full(frames=6).to(device)
    sum_Generate_net = sum(p.numel() for p in Generate_net.parameters())* 4 / (1024 ** 2)
    print("Generate_net total number of parameters is {}MB".format(round(sum_Generate_net,2 )))
    
    Spatial_dis = Spatial().to(device)
    Temporal_dis = Temporal().to(device)
    Spatial_dis_sum = sum(p.numel() for p in Spatial_dis.parameters())* 4 / (1024 ** 2)
    
    Temporal_dis_sum = sum(p.numel() for p in Temporal_dis.parameters())* 4 / (1024 ** 2)
    
    cycle_lengths=[10000000000000]
    f_start = [1e-6]
    f_max = [1.]
    f_min = [1.]
    warm_up_steps = [1000]
    
    Generate_net_optim = optim.Adam(Generate_net.parameters(),lr=2e-4,betas=(0.0, 0.999))
    
    Spatial_dis_optim = optim.Adam(Spatial_dis.parameters(),lr=2e-5,betas=(0.0,0.999))
    
    Temporal_dis_optim = optim.Adam(Temporal_dis.parameters(),lr=2e-5,betas=(0.0,0.999))
    
    
    scheduler_Gen = lr_scheduler.LambdaLR(Generate_net_optim, lr_lambda=LambdaLinearScheduler(warm_up_steps=warm_up_steps, f_max=f_min, f_min=f_max, f_start=f_start,
                                                                                          cycle_lengths=cycle_lengths).schedule)
    
    scheduler_Spa = lr_scheduler.LambdaLR(Spatial_dis_optim, lr_lambda=LambdaLinearScheduler(warm_up_steps=warm_up_steps, f_max=f_min, f_min=f_max, f_start=f_start,
                                                                                          cycle_lengths=cycle_lengths).schedule)
    
    scheduler_Tem = lr_scheduler.LambdaLR(Temporal_dis_optim, lr_lambda=LambdaLinearScheduler(warm_up_steps=warm_up_steps, f_max=f_min, f_min=f_max, f_start=f_start,
                                                                                          cycle_lengths=cycle_lengths).schedule)
    
    print("Spatial_dis_sum total number of parameters is {}MB \nTemporal_dis_sum total number of parameters is {}MB".format(round(Temporal_dis_sum,2),round(Spatial_dis_sum,2)))
    
    

    
    
    
    print("ready dataset")
    
    mydataset = MyDataset("/example")
    
    dataloader =DataLoader(mydataset, batch_size=2, shuffle=False,drop_last=True)
    
    # choose generator funciton
    
    Generator_loss_skillful_f = Generator_loss_skillful().to(device)
    DiscriminatorLoss_hinge_f = DiscriminatorLoss_hinge().to(device)
    # x,y = next(iter(dataloader))
    for epoch in range(500):
        for step,dataes in enumerate(tqdm(dataloader)):
            ### Training the Generate #####
            x,y = dataes
            x=x.to(device)
            y=y.to(device)
            Generate_net.train()
            Generate_net_optim.zero_grad()
            gen_out = Generate_net(torch.squeeze(x))# The out's shape is -> b t c w h
            gen_out_copy = gen_out.clone()
            tem_out = Temporal_dis(gen_out)
            spa_loss = Spatial_dis(gen_out)
            dis_loss = DiscriminatorLoss_hinge_f(spa_loss,True)+DiscriminatorLoss_hinge_f(tem_out,True)
            Gen_loss = Generator_loss_skillful_f(y,gen_out,dis_loss)
            Gen_loss.backward()
            Generate_net_optim.step()
            
            
            write.add_scalar("Generate loss",Gen_loss.item(),step*epoch+step)
            
            ### Train the Discriminatores
            """
            traing the spatialdiscriminatores
            """
            Spatial_dis_optim.zero_grad()
            spa_loss_fake = Spatial_dis(gen_out_copy.detach())
            spa_loss_real = Spatial_dis(y)
            spa_loss_traing = DiscriminatorLoss_hinge_f(spa_loss_fake,False)+DiscriminatorLoss_hinge_f(spa_loss_real,True)
            spa_loss_traing.backward()
            Spatial_dis_optim.step()
            
            write.add_scalar("Spatial loss",spa_loss_traing.item(),step*epoch+step)
            
            
            """
            traing the Temporaldiscriminatores
            """
            
            Temporal_dis_optim.zero_grad()
            spa_loss_fake = Temporal_dis(torch.cat([x,gen_out_copy.detach()],dim=1))
            spa_loss_real = Temporal_dis(torch.cat([x,y],dim=1))
            Tem_loss_traing = DiscriminatorLoss_hinge_f(spa_loss_fake,False)+DiscriminatorLoss_hinge_f(spa_loss_real,True)
            Tem_loss_traing.backward()
            Temporal_dis_optim.step()
            
            write.add_scalar("Temporal loss",Tem_loss_traing.item(),step*epoch+step)
            #write.add_image("Sampling during training",gen_out_copy.detach(),step*epoch+step)
            write.add_scalar("Gen_lr",Generate_net_optim.param_groups[0]['lr'],step*epoch+step)
            write.add_scalar("Spa_lr",Spatial_dis_optim.param_groups[0]['lr'],step*epoch+step)
            write.add_scalar("Tem_lr",Temporal_dis_optim.param_groups[0]['lr'],step*epoch+step)
            scheduler_Gen.step()
            scheduler_Spa.step()
            scheduler_Tem.step()
        if epoch%10==0:
            print(x.shape,gen_out_copy.shape)
            rainprint(torch.concat([x,gen_out_copy.detach()],dim=1),"tu2net/Sampe_reslut_during_training/{}.jpg".format(epoch))
            torch.save(Generate_net.state_dict(),"tu2net/Generate_pth/gen-{}.pth".format(epoch))
            torch.save(Temporal_dis.state_dict(),"tu2net/Tem_pth/gen-{}.pth".format(epoch))
            torch.save(Spatial_dis.state_dict(),"tu2net/Spa_pth/gen-{}.pth".format(epoch))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


if __name__ == '__main__':  
    train()