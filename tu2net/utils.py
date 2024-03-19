from matplotlib import pyplot as plt
import torch
import einops
import numpy

def rainprint(x:torch.tensor,img_path="result/sample.jpg",remove_background=False,dpi=300,vmax=10,vmin = 0,cmap = "jet",renormalization=True):
    # x = x.detach().cpu().numpy()
    # print(x.shape)
    out = einops.rearrange(torch.squeeze(x.detach().cpu()).numpy(),"b t w h -> (b w) (t h)")
    if renormalization:
        out = out*22.0
    plt.figure(figsize=(9, 9))
    plt.axes()
    plt.axis('off')
    plt.imshow(out, vmax=vmax, vmin=vmin, cmap=cmap)
    
    if remove_background:
        out[out<0.01] = numpy.nan
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    

    
    
    
    