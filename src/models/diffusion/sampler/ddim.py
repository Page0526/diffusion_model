import torch
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion.sampler.base import BaseSampler, expand_dim_like


class DDIMSampler(BaseSampler):

    def __init__(self,
                 n_train_steps: int = 1000,
                 n_infer_steps: int = 50,
                 beta_schedule: str = 'cosine',
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 clip_denoised: bool = True,
                 set_final_alpha_to_one: bool = True) -> None:
        """
        n_train_steps:
        n_infer_Steps:
        beta_schedule:
        beta_start:
        beta_end:
        clip_denoised:
        """

        super().__init__(n_train_steps, n_infer_steps, beta_schedule,
                         beta_start, beta_end,
                         clip_denoised, set_final_alpha_to_one)

    def reverse_step(self,
                     model_output: Tensor,
                     t: Tensor,
                     xt: Tensor,
                     noise: Tensor | None = None,
                     repeat_noise: bool = False) -> Tensor:

        # ddpm so: t_prev != t - 1
        t_prev = t - self.n_train_steps // self.n_infer_steps

        alpha_bar = self.alpha_bar[t].flatten()
        alpha_bar = expand_dim_like(alpha_bar, xt)

        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].flatten()
        sqrt_one_minus_alpha_bar = expand_dim_like(sqrt_one_minus_alpha_bar,
                                                    xt)

        alpha_bar_prev = torch.where(t_prev >= 0, self.alpha_bar[t_prev],
                                     self.final_alpha_bar).flatten()
        alpha_bar_prev = expand_dim_like(alpha_bar_prev, xt)

        # compute x_0 of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        x0_pred = (xt - sqrt_one_minus_alpha_bar * model_output) / (alpha_bar
                                                                     **0.5)

        if self.clip_denoised:
            x0_pred.clamp_(-1.0, 1.0)

        # sigma = 0 in formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        mean = alpha_bar_prev**0.5 * x0_pred + (
            1 - alpha_bar_prev)**0.5 * model_output

        # ddim: noise = 0
        return mean


if __name__ == "__main__":
    sampler = DDIMSampler()
    print(sampler.timesteps)