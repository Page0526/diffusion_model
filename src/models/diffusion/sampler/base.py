from typing import Literal, TypeAlias, Union, List, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from src.models.unet.unet import UNet

def expand_dim_like(x: Tensor, y: Tensor):
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x

def noise_like(shape: List[int], device: torch.device, repeat: bool = False):
    '''
    create a tensor with the same dimension as shape, except that batch dimension (first dim) is set to 1
    create a tensor where all batch elements have same noise
    *((1,) * (len(shape) - 1)) ensures the other dimensions remain unchanged while repeating along the batch dimension
    '''
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1, ) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def beta_scheduler(n_steps: int = 1000,
                    beta_schedule: str = "linear",
                    beta_start: float = 1e-4,
                    beta_end: float = 0.02,
                    max_beta: float = 0.999,
                    s: float = 0.008, # for improvedDDPM
                    device: Union[torch.device, str] | None = None,
                    ):
        
    if beta_schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32, device=device)

    elif beta_schedule == "cosine":
        f = [math.cos((t / n_steps + s) / (1 + s) * (math.pi / 2))**2 for t in range(n_steps + 1)]

        beta = []
        for t in range(n_steps):
            beta.append((min(1 - f[t + 1] / f[t], max_beta)))
    
        return Tensor(beta)
        
    assert beta.shape == (n_steps, ), f"not enough {n_steps} steps"
    return beta
    
class BaseSampler(nn.Module):
    def __init__(self,
                 n_train_steps: int,
                 n_infer_steps: int,
                 beta_schedule: str = 'linear',
                 beta_start: float=1e-4,
                 beta_end: float=2e-2,
                 clip_denoised: bool = True,
                 set_final_alpha_to_one: bool = True,
                 ):
        super().__init__()
        self.n_train_steps = n_train_steps
        self.clip_denoised = clip_denoised
        self.set_timesteps(n_infer_steps)
        self.register_schedule(beta_schedule=beta_schedule, 
                               beta_start=beta_start, 
                               beta_end=beta_end, 
                               n_steps=n_train_steps, 
                               set_final_alpha_to_one=set_final_alpha_to_one)

    # this function will automatically send betas, alphas to same device
    def register_schedule(
        self,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        set_final_alpha_to_one: bool = True,
        n_steps: int = 1000,
    ):
        betas = beta_scheduler(n_steps, beta_schedule, beta_start,
                                       beta_end)

        alpha = 1.0 - betas
        alpha_bar = torch.cumprod(alpha, dim=0)

        final_alpha_bar = Tensor(
            [1.0]) if set_final_alpha_to_one else alpha_bar[0]

        # automatic to device
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("final_alpha_bar", final_alpha_bar)


    def set_timesteps(self, n_infer_steps: int):
        if n_infer_steps > self.n_train_steps:
            raise ValueError(
                f"Number of inference steps ({n_infer_steps}) cannot be large than number of train steps ({self.n_train_steps})"
            )
        else:
            self.n_infer_steps = n_infer_steps
        
        self.timesteps = (torch.linspace(0, self.n_train_steps - 1, n_infer_steps).flip(0).to(torch.int64))
        
    def step(self, x0: Tensor,
             t: Tensor, noise: Tensor | None = None) -> Tensor:
        '''
        perform a single step of diffusion process
        sample from q(x_t|x_0)
        Args:
            x0: samples with noiseless (input)
            t: timesteps in diffusion process
            noise: noise tensor for curr t
        Returns:
            xt (Tensor): noisy sample
        '''
        if noise is None:
            # randn_like != rand_like, gaussian vs uniform
            noise = torch.randn_like(x0)
        else:
            assert noise.shape == x0.shape, 'shape not match'
            noise = noise.to(x0.device)

        sqrt_alpha_prod = self.sqrt_alpha_bar[t].flatten()
        sqrt_alpha_prod = expand_dim_like(sqrt_alpha_prod, x0)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].flatten()
        sqrt_one_minus_alpha_bar = expand_dim_like(sqrt_one_minus_alpha_bar, x0)

        mean = sqrt_alpha_prod * x0
        std = sqrt_one_minus_alpha_bar

        return mean + std * noise

    def get_variance(self,
                     t: Tensor,
                     t_prev: Tensor,
                    ):
        alpha_bar = self.alpha_bar[t]
        alpha_bar_prev = torch.where(t_prev >= 0, self.alpha_bar[t_prev], self.final_alpha_bar)

        beta = 1 - alpha_bar / alpha_bar_prev

        var = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta

        # from IPython import embed
        # embed
        # NOTE: what does this do?
        var = torch.clamp(var, min=1e-20)

        return var

    def reverse_step(self,
                     model_output: Tensor,
                     t: Tensor,
                     xt: Tensor,
                     eta: float = 0.0, # DDPM - eta = 1, DDIM - eta = 0
                     noise: Tensor | None = None,
                     repeat_noise: bool = False) -> Tensor:
        '''
        predict sample at previous timestep by reversing SDE
        Args:
            model_output: output of denoise net
            t: timestep in the diffusion chain
            xt: cur instance of sample being created by diffusion forward process
            noise
            repeat_noise
        Returns:
            x_{t-1} sample at previous timestep
        '''
        t_prev = t - self.n_train_steps // self.n_infer_steps

        alpha_bar = self.alpha_bar[t].flatten()
        alpha_bar = expand_dim_like(alpha_bar, xt)

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].flatten()
        sqrt_one_minus_alpha_bar = expand_dim_like(sqrt_one_minus_alpha_bar, xt)

        alpha_bar_prev = torch.where(t_prev>=0, self.alpha_bar[t_prev], self.final_alpha_bar).flatten()
        alpha_bar_prev = expand_dim_like(alpha_bar_prev, xt)

        x0_pred = (xt - sqrt_one_minus_alpha_bar * model_output) / (alpha_bar**0.5)

        # from IPython import embed
        # embed
        # NOTE: what does this do?
        if self.clip_denoised:
            x0_pred.clamp_(-1.0, 1.0)

        if noise is None:
            if repeat_noise: 
                noise = noise_like(xt.shape, device=xt.device, repeat=repeat_noise)
            else:
                noise = torch.randn_like(xt)
        else:
            assert noise.shape == xt.shape, 'shape not match'
            noise = noise.to(xt.device)
        
        var = torch.zeros_like(model_output)
        # t = 0 (the last step reverse) -> not add noise
        # from IPython import embed
        # embed()
        ids = t > 0
        var_dtype = xt.dtype
        var[ids] = expand_dim_like(self.get_variance(t, t_prev)[ids].to(var_dtype), xt)
        std = var**0.5

        std = std * eta

        mean = alpha_bar_prev**0.5 * x0_pred + (1 - alpha_bar_prev -
                                                var)**0.5 * model_output

        return mean + std * noise



