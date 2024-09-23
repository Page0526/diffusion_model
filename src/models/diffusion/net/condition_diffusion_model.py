from typing import List, Tuple, Dict
import torch
from torch import nn
from torch import Tensor
import math
from tqdm import tqdm
# help set up and find dir of project
import pyrootutils
from src.models.unet.unet_condition import ConditionalUNet
from src.models.diffusion.net.diffusion_model import DiffusionModel
from src.models.diffusion.sampler.base import BaseSampler, noise_like
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ConditionDiffusionModel(DiffusionModel):
    """
    ### Condition Diffusion Model
    """

    def __init__(
        self,
        denoise_net: ConditionalUNet,
        sampler: BaseSampler,
        label_embedder: nn.Module = None,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
    ):
        super().__init__(n_train_steps=n_train_steps,
                         sampler=sampler,
                         denoise_net=denoise_net,
                         img_dims=img_dims)
        self.label_embedder = label_embedder

    def get_label_embedding(self, label: torch.Tensor):
        return self.label_embedder(label)

    def forward(self,
                x0: Tensor,
                sample_steps: Tensor | None = None,
                noise: Tensor | None = None,
                cond: Tensor | None = None,
                ):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        # Forward pass
        
        # start with x0
        if sample_steps is None:
            sample_steps = torch.randint(
                0,
                self.n_train_steps,
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
        
        xt = self.sampler.step(x0=x0, t=sample_steps, noise=noise)

        # calculate predicted noise        
        pred_noise = self.denoise_net(xt, sample_steps - 1, cond=cond)
        
        return pred_noise, noise

    @torch.no_grad()
    def sample(self,
               cond: Tensor,
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
           
            model_output = self.denoise_net(xt ,t, cond=cond)

            xt = self.sampler.reverse_step(model_output=model_output, t=t, xt=xt, noise=noise,
                                           repeat_noise=repeat_noise)

        return xt
    