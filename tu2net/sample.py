from matplotlib import pyplot as plt
from TU2Net import Generator_full
import torch
import numpy
import einops
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
# pth_path = "/media/ybxy/c89da59f-580c-440d-bab8-554bd51bb407/tu2NET/Tu2net/tu2net.pth"
example_pth = ""
class Get_tager_sample(Dataset):

    def __init__(self,path):
        self.img_path = os.listdir(path)
        self.path = path
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        radar = numpy.load(os.path.join(self.path,img_name))
        # Mytrainsform(radar)
        radar =torch.from_numpy(radar)

        tagert = (radar[0:4,:,:,:]- 0.4202) / 0.8913
        sample = (radar[4:10,:,:,:]- 0.4202) / 0.8913
        return tagert,sample

    def __len__(self):
        return len(self.img_path)


data = Get_tager_sample("tu2net/example")
val_t = DataLoader(data, batch_size=2,shuffle=False,drop_last=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Generator_full(device='cuda:0').to(device)
# net.load_state_dict(torch.load(pth_path))
net.eval()
x,y = next(iter(val_t))
with torch.no_grad():
    out = net(torch.squeeze(x.to(device))).cpu().detach()
    print(out.shape)


out = einops.rearrange(torch.squeeze(out).numpy(),"b t w h -> (b w) (t h)")
lable = einops.rearrange(torch.squeeze(y).numpy(),"b t w h -> (b w) (t h)" )


plt.figure(figsize=(9, 9))
plt.axes()
plt.axis('off')
im1 = Image.fromarray(out.astype('uint8'))
im2 = Image.fromarray(lable.astype('uint8'))
plt.imshow(im1, vmax=10, vmin=0, cmap="jet")
plt.savefig("sample.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
plt.imshow(im2, vmax=10, vmin=0, cmap="jet")
plt.savefig("lable.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
    
    




