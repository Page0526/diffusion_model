from typing import List, Tuple, Dict
import torch
from torch import nn
from torch import Tensor
import math
from tqdm import tqdm
# help set up and find dir of project
import pyrootutils
from src.models.unet.unet_condition import ConditionalUNet
from src.models.diffusion import DiffusionModel
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ConditionDiffusionModel(DiffusionModel):
    """
    ### Condition Diffusion Model
    """

    def __init__(
        self,
        timesteps: int,
        denoise_net: ConditionalUNet,
        label_embedder: nn.Module = None,
        image_channels: int | None = 1,
        device: torch.device = torch.device('cuda'),
        eta: int | None = 1, # eta = 1 : DDPM, eta = 0: DDIM
        infer_steps: int | None = 1, # ddpm
    ):
        super().__init__(timesteps, denoise_net, image_channels, device, eta, infer_steps)
        self.label_embedder = label_embedder

    def get_label_embedding(self, label: torch.Tensor):
        return self.label_embedder(label)

    def forward(self, x0: Tensor, sample_steps: Tensor | None = None, noise: Tensor | None = None, label: Tensor | None = None):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        if sample_steps is None:
            sample_steps = torch.randint(0, self.timesteps, [x0.shape[0]], device=x0.device)
        else:
            sample_steps = sample_steps.to(x0.device)

        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar.to(x0.device)[sample_steps - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred_noise = self.denoise_net(torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, sample_steps - 1, label)

        return pred_noise, noise

    @torch.no_grad()
    def sample(self, n_samples: int | None = 1,
               img_size=(32, 32),
               device: torch.device = torch.device('cuda'),
               labels=None,
               use_tqdm=False):
        x = torch.randn((n_samples, self.image_channels, img_size[0], img_size[1]), device=device)
        tau = self.timesteps // self.infer_steps
        ddim_timesteps = list(range(1, self.timesteps, tau))

        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.timesteps, 1, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=device) * t

            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_prev_t = self.alpha_bar[t - 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps = self.denoise_net(x, t - 1, labels)

            x0_t = (x - eps * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
            c1 = self.eta * torch.sqrt((1 - alpha_bar_t / alpha_bar_prev_t) * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t))
            c2 = torch.sqrt((1 - alpha_bar_prev_t) - c1 ** 2)
            x = torch.sqrt(alpha_bar_prev_t) * x0_t + c1 * z + c2 * eps

        return x


if __name__ == "__main__":
    _ = DiffusionModel(1000)