from typing import Tuple, List, Dict
import torch
from torch import nn
from torch import Tensor
import math
from tqdm import tqdm
# help set up and find dir of project
import pyrootutils

from src.utils.utils import c2r, r2c

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# follow https://github.com/bo-10000/MoDL_PyTorch   
class MoDLModel(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        
        x_k = x0.clone()
        for k in range(self.k_iters):
            #dw 
            z_k = self.dw(x_k) # (2, nrow, ncol)
            #dc
            x_k = self.dc(z_k, x0, csm, mask) # (2, nrow, ncol)
        return x_k

def conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(2, 64)

        for _ in range(n_layers - 2):
            layers += conv_block(64, 64)

        layers += nn.Sequential(
            nn.Conv2d(64, 2, 3, 1),
            nn.BatchNorm2d(2)
        )

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x # (2, row, col)
        return self.nw(x) + idt # (2, row, col)
        
class myAtA(nn.Module):
    def __init__(self, csm, mask, lam):
        super().__init__()
        self.csm = csm
        self.mask = mask
        self.lam = lam

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_coil = self.csm * im # split coil images (B x ncoil x nrow x ncol)
        k_full = torch.fft.fft2(im_coil, norm='ortho') # convert into k-space 
        k_u = k_full * self.mask # undersampling
        im_u_coil = torch.fft.ifft2(k_u, norm='ortho') # convert into image domain
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=1) # coil combine (B x nrow x ncol)
        return im_u + self.lam * im
    
def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec