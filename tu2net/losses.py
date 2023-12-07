import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet34
import einops 
#Here we provide multiple reconstruction losses and regularization methods



class Generator_loss_skillful(nn.Module):
    """
    https://www.nature.com/articles/s41586-021-03854-z
    """
    def __init__(self):
        super(Generator_loss_skillful, self).__init__()
        
    def forward(self, org_img, pre_img, loss_dis):
        weights = torch.clip(org_img, 0.0, 22.0)
        loss = torch.mean(torch.abs(org_img - pre_img) * weights)
        x = loss_dis+loss
        return x





class Generator_loss_reconstruction_with_resnet34(nn.Module):
    """
    Using resnet34 as perceptual loss
    """
    def __init__(self) -> None:
        super().__init__()
        resnet_34 = resnet34(pretrained=True)
        self.model = nn.Sequential(*list(resnet_34.children())[:-1])
        for p in self.model.parameters():
            p.requires_grad = False
        
    def forward(self,org_img,pre_img,loss_dis,reconstruction_distance = "l2"):
        org_img = einops.repeat(org_img,"b t c w h -> (b t) (c a) w h",a = 3)
        pre_img = einops.repeat(pre_img,"b t c w h -> (b t) (c a) w h",a = 3)
        org_emb = self.model(org_img)
        pre_emd = self.model(pre_img)
        emd_distance = nn.functional.mse_loss(org_emb,pre_emd)
        if reconstruction_distance == "l1":
            reconstruction_loss =  torch.abs(org_img - pre_img).mean()
        else:
            reconstruction_loss = nn.functional.mse_loss(org_emb,pre_emd)
        # print(reconstruction_loss)
        total_loss = reconstruction_loss*0.7+ emd_distance*0.3 + loss_dis
        
        return total_loss


class DiscriminatorLoss_hinge(nn.Module):
    """
    Discriminator hinge loss.
    """
    def __init__(self):
        super().__init__()
    def forward(self, T_dis_org,org: bool) :  # dim修改
        if org is True:
            loss = nn.functional.relu(1 - T_dis_org)
        else:
            loss =nn.functional.relu(1 + T_dis_org)
        
        return torch.mean(loss)

class DiscriminatorLoss_BCE(nn.Module):
    """
    Discriminator BCE loss.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, T_dis_org,org: bool):
        if org is True:
            garget = torch.ones_like(T_dis_org)
        else:
            garget = torch.zeros_like(T_dis_org)
        return nn.functional.binary_cross_entropy_with_logits(garget,T_dis_org)    
            
        
        
        
        


x = torch.rand(size=(4,6,1,256,256))
y = torch.rand(size=(4,6,1,256,256))

g_resnet = Generator_loss_reconstruction_with_resnet34()

print(g_resnet(x,y,torch.tensor(1.0)))

DiscriminatorLoss_hinge(dis_real_out,True)
DiscriminatorLoss_hinge(dis_pre_out,False)

        
        
        
            
    
    





