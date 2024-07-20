from typing import Tuple, List, Dict

import torch
import random
import math
from torch import Tensor
import pyrootutils
import torch.nn as nn
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.unet.net import Unet

class DiffusionModel(nn.Module):

    def __init__(
        self, 
        denoise_net: Unet,
        sample: , 
        timesteps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.img_dims = img_dims
        self.denoise_net = denoise_net
        self.sampler = sampler

    def train_step(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        # Forward pass
        
        # start with x0
        x0 = sample_batch(train_dataloader, self.device)
        # print(x0.shape) # [64, 1, 32, 32] 
        # Uniform distribution from 1 to T         
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device,
                          dtype=torch.long)
        # Normalized distribution ~ N(0, I)        
        eps = torch.randn_like(x0)

        # Take one gradient descent step on
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # epsilon predicted = noise ?         
        eps_predicted = self.function_approximator(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t - 1)
        # can it be replace by different loss functon ?         
        loss = nn.functional.mse_loss(eps, eps_predicted)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                 use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """
        # Normalized distribution ~ N(0, I)         
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * self.function_approximator(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        # print(f"Diffusion-sampling-x.shape:{x.shape}")
        return x
    
    # cosine scheduler function, s = small offset prevent beta_t from being too small near t = 0    
    def cosine_variance_schedule(self, s=0.008):
        step = torch.linspace(0, self.T, steps=self.T+1,dtype=torch.float32)
        temp_sum = (step/self.T + s)/(1 + s) * (math.pi*0.5)
        f_t = torch.cos(temp_sum)**2
        beta = torch.clip(1.0-f_t[1:]/f_t[self.T],0.0,0.999)
        return beta

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "diffusion" / "net")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="diffusion_model.yaml")
    def main(cfg: DictConfig):
        cfg['n_train_steps'] = 1000
        cfg['img_dims'] = [1, 32, 32]
        cfg['sampler']['n_train_steps'] = 1000
        print(cfg)

        diffusion_model: DiffusionModel = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, cfg['n_train_steps'], (2, ))

        print('*' * 20, ' DIFFUSION MODEL ', '*' * 20)

        print('=' * 15, ' forward process ', '=' * 15)
        print('Input:', x.shape)
        xt = diffusion_model.sampler.step(x, t)
        pred, target = diffusion_model(x, t)  # with given t
        pred, target = diffusion_model(x)  # without given t
        print('xt:', xt.shape)
        print('Prediction:', pred.shape)
        print('Target:', target.shape)

        print('=' * 15, ' reverse process ', '=' * 15)
        gen_samples = diffusion_model.sample(num_sample=2, prog_bar=True)
        print(len(gen_samples), gen_samples[0].shape)

    main()