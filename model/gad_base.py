# modified model
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from random import randrange
from re import I

INPUT_DIM = 4
FEATURE_DIM = 64

class GADBase_LST(nn.Module):
    def __init__(
            self, feature_extractor='none',
            Npre=8000, Ntrain=1024, 
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        print(feature_extractor)
 
        if feature_extractor=='none': 
            print('RGB version')
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = nn.Parameter(torch.log(torch.tensor(0.03)))
            #self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            self.feature_extractor = self.create_unet_feature_extractor()
            self.logk = nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

    def create_unet_feature_extractor(self):
        unet = smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM)
        
        class UNetWrapper(nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, x):
                original_size = x.shape[2:]
                x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
                x = self.unet(x)
                x = F.adaptive_avg_pool2d(x, original_size)
                return x

        return UNetWrapper(unet).cuda()

    def forward(self, sample, train=False, deps=0.0):
        guide, source, mask_lr = sample['image'], sample['source'], sample['mask_lr']

        if source.min() < deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarily shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        if shifted:
            y_pred -= deps

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1) 
        else:
            guide_feats = self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1))
        
        cv, ch = c(guide_feats, K=K)

        if self.Npre > 0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                for t in range(Npre):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        if self.Ntrain > 0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        return img, {"cv": cv, "ch": ch}

# The rest of the functions (c, g, diffuse_step, adjust_step) remain unchanged

# @torch.jit.script
#I: guide_features
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th

    #print("Diffusion step - I range:", I.min().item(), I.max().item())
    #print("Diffusion step - tv range:", tv.min().item(), tv.max().item())
    #print("Diffusion step - th range:", th.min().item(), th.max().item())

    
    
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    #print("Adjust step - img_ss range:", img_ss.min().item(), img_ss.max().item())

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    #print("Adjust step - ratio_ss range:", ratio_ss.min().item(), ratio_ss.max().item())
    #print("Adjust step - any infinite values in ratio_ss?", torch.isinf(ratio_ss).any().item())
    ratio_ss[mask_inv] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    # ratio = torch.sqrt(ratio)
    # img = img * R
    #print("Adjust step - final ratio range:", ratio.min().item(), ratio.max().item())
    return img * ratio 