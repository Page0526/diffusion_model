import torch
from torch import nn
from torch import Tensor
import math
from tqdm import tqdm
# help set up and find dir of project
import pyrootutils
from src.models.unet.unet import UNet
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DiffusionModel(nn.Module):

    def __init__(self,
                 timesteps: int,
                 denoise_net: UNet,
                 image_channels: int | None = 1,
                 device: torch.device = torch.device('cuda'),
                 eta: int | None = 1, # eta = 1 : DDPM, eta = 0: DDIM
                 infer_steps: int | None = 1 # ddpm
                 ):
        super().__init__()
        self.timesteps = timesteps # number of time steps
        self.device = device
        self.denoise_net = denoise_net.to(device) # Unet (neural network)
        # linear schedule   
        self.beta = torch.linspace(1e-4, 0.02, self.timesteps).to(device)
        # self.beta = self.cosine_variance_schedule()
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.image_channels = image_channels
        self.eta = eta
        self.infer_steps = infer_steps

    def forward(self,
                x0: Tensor,
                sample_steps: Tensor | None = None,
                noise: Tensor | None = None):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        # Forward pass
        
        # start with x0
        if sample_steps is None:
            sample_steps = torch.randint(
                0,
                self.timesteps,
                [x0.shape[0]],
                device=x0.device, 
            )
        else:
            assert sample_steps.shape[0] == x0.shape[0], 'batch_size not match'
            sample_steps = sample_steps.to(x0.device)

        # # Uniform distribution from 1 to T         
        # t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)

        # Normalized distribution ~ N(0, I) 
        if noise is None:       
            noise = torch.randn_like(x0)

        # Take one gradient descent step on
        alpha_bar_t = self.alpha_bar.to(x0.device)[sample_steps - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # calculate predicted noise        
        pred_noise = self.denoise_net(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, sample_steps - 1)

        return pred_noise, noise

    @torch.no_grad()
    def sample(self,
               n_samples: int | None = 1,
               img_size=(32, 32),
               device: torch.device = torch.device('cuda'),
               use_tqdm=False):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """
        # Normalized distribution ~ N(0, I)     
        x = torch.randn((n_samples, self.image_channels, img_size[0], img_size[1]), device=device)

        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_prev_t = self.alpha_bar[t - 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps = self.denoise_net(x, t - 1)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.timesteps, 1, -self.infer_steps)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=device) * t
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_prev_t = self.alpha_bar[t - self.infer_steps - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps = self.denoise_net(x, t - 1)
            # DDIM
            if self.eta == 0:
                x = (torch.sqrt(alpha_bar_prev_t) * (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t))
            # DDPM
            else:
                sigma_t = self.eta * torch.sqrt((1 - alpha_bar_prev_t) * (1 - alpha_bar_t) / (1 - alpha_bar_t))
                noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                img = (torch.sqrt(alpha_bar_prev_t) * (img - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)) + sigma_t * noise

        return x
     
    # cosine scheduler function, s = small offset prevent beta_t from being too small near t = 0    
    def cosine_variance_schedule(self, s=0.008):
        step = torch.linspace(0, self.T, steps=self.T+1,dtype=torch.float32)
        temp_sum = (step/self.T + s)/(1 + s) * (math.pi*0.5)
        f_t = torch.cos(temp_sum * temp_sum)
        beta = torch.clip(1.0-f_t[1:]/f_t[self.T],0.0,0.999)
        return beta.to(self.device)
    
if __name__ == "__main__":
    _ = DiffusionModel(1000)