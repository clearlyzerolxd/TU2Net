import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet34

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
    def __init__(self) -> None:
        super().__init__()
        resnet_34 = resnet34(pretrained=True)
        self.model = nn.Sequential(*list(resnet_34.children())[:-1])
        for p in self.model.parameters():
            p.requires_grad = False
        
    def forward(self,org_img,pre_img,loss_dis):
        self.model(org_img)
        self.model(pre_img)
        
        

        
        
        
            
    
    





