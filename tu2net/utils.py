from matplotlib import pyplot as plt
import torch
import einops
import numpy

def rainprint(x:torch.tensor,img_path="result/sample.jpg",remove_background=False,dpi=300):
    out = einops.rearrange(torch.squeeze(x).numpy(),"b t w h -> (b w) (t h)")
    plt.figure(figsize=(9, 9))
    plt.axes()
    plt.axis('off')
    plt.imshow(out, vmax=10, vmin=0, cmap="jet")
    
    if remove_background:
        out[out<0.01] = numpy.nan
    
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    

    
    
    
    