import torch
from torch import nn
from torch import Tensor
import math
from tqdm import tqdm
# help set up and find dir of project
import pyrootutils
from typing import Tuple, List, Dict
from src.models.unet.unet import UNet
from src.models.diffusion.sampler import BaseSampler
from src.models.diffusion.sampler import noise_like

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DiffusionModel(nn.Module):

    def __init__(self,
                 denoise_net: UNet,
                 sampler: BaseSampler,
                 n_train_steps: int = 1000,
                 img_dims: Tuple[int, int, int] = [1, 32, 32],
                 ):
        super().__init__()
        self.n_train_steps = n_train_steps # number of time steps
        self.denoise_net = denoise_net # Unet (neural network)
        self.sampler = sampler
        self.img_dims = img_dims


    def forward(self,
                x0: Tensor,
                sample_steps: Tensor | None = None,
                noise: Tensor | None = None,
                ):
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

        # Normalized distribution ~ N(0, I) 
        if noise is None:       
            noise = torch.randn_like(x0)
        else:
            assert noise.shape == x0.shape, 'shape not match'
            noise = noise.to(x0.device)
        
        xt = self.sampler.step(x0, sample_steps, noise)

        # calculate predicted noise        
        pred_noise = self.denoise_net(xt, sample_steps - 1)

        return pred_noise, noise

    @torch.no_grad()
    def sample(self,
               sample_steps: Tensor | None = None,
               n_samples: int | None = 1,
               xt: Tensor | None =  None,
               noise: Tensor | None = None,
               repeat_noise: bool = False,
               device: torch.device = torch.device('cuda'),
               prog_bar: bool = False,
               ):
        """_summary_
        ### reverse diffusion process
        Args:
            xt (Tensor | None, optional): _description_. Defaults to None. -> x_last: x_{tau_s} - if not provided random noise will be used
            sample_steps (Tensor | None, optional): _description_. Defaults to None.
            n_samples (int, optional): _description_. Defaults to 1.
            noise (Tensor | None, optional): _description_. Defaults to None.
            repeat_noise (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to torch.device('cuda').
        Returns:
            List[Tensor]: _description_
        """
        # xt ~ N(0, I)
        if xt is None:
            assert n_samples, 'n_sample is None'

            xt = noise_like([n_samples] + list(self.img_dims),
                             device=device,
                             repeat=repeat_noise)
        else:
            assert xt.shape[1:] == self.img_dims, 'shape of image is not match'
            xt = xt.to(device)
        
        if sample_steps is None:
            sample_steps = tqdm(
                self.sampler.timesteps) if prog_bar else self.sampler.timesteps
        else:
            assert sample_steps.shape[0] == xt.shape[
                0], 'batch of sample_steps and xt not match'
            sample_steps = tqdm(sample_steps) if prog_bar else sample_steps

        for i, t in enumerate(sample_steps):

            t = torch.full((xt.shape[0], ),
                           t,
                           device=device,
                           dtype=torch.int64)

            model_output = self.denoise_net(xt ,t)


            xt = self.sampler.reverse_step(model_output, t, xt, noise,
                                           repeat_noise)

        return [xt]
    
if __name__ == "__main__":
    _ = DiffusionModel(1000)