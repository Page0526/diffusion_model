from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# follow https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddim.py
class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module, # unet model
        betas: Tuple[float, float],
        n_T: int, # number of timesteps
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return loss between pred noise and noise
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            # x = mean + sigma * z
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

# https://arxiv.org/abs/2010.02502
class DDIM(DDPM):
    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            eta: float,
            n_T: int, # number of timesteps
            criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDIM, self).__init__(eps_model, betas, n_T, criterion)
        self.eta = eta

    # modified from https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L10-L32
    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 1, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1))
            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()
            c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i

# follow https://github.com/paulaceccon/deep-learning-studies/blob/main/notebooks/generative_models/diffusion_models/ddim.py
class DDIM(DDPM):
    """
    DDPM Sampling.

    Args:
        eps_model: A neural network model that predicts the noise term given a tensor.
        betas: A tuple containing two floats, which are parameters used in the DDPM schedule.
        eta: Scaling factor for the random noise term.
        n_timesteps: Mumber of timesteps in the diffusion process.
        criterion: Loss function.
    """

    def __init__(
            self,
            eps_model: nn.Module,
            betas: tuple[float, float],
            eta: float,
            n_timesteps: int,
            criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDIM, self).__init__(eps_model, betas, n_timesteps, criterion)
        self.eta = eta

    def sample(self, n_samples: int, size: torch.Tensor, device: str) -> torch.Tensor:
        # Initialize x_i with random noise from a standard normal distribution
        # x_i corresponds to x_T in the diffusion process, where T is the total number of timesteps
        x_i = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1)

        # Iterate backwards through the timesteps from n_timesteps to 1 -> reverse process
        for i in range(self.n_timesteps, 1, -1):
            # Eq. (10)
            # Sample additional random noise z, unless i is 1 (in which case z is 0, i.e., no additional noise)
            z = torch.randn(n_samples, *size).to(device) if i > 1 else 0  # z ~ N(0, 1) for i > 1, else z = 0
            
            # eps = self.eps_model(x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1))
            # Predict the noise eps to be removed at the current timestep, using the eps_model
            # The current timestep i is normalized by n_timesteps and replicated for each sample
            eps = self.eps_model(x_i, torch.tensor(i / self.n_timesteps).to(device).repeat(n_samples, 1))

            # predicted x_0
            # Calculate the predicted x0 (original data) at timestep 'i'
            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()

            # Compute coefficients for the DDIM sampling process.
            # c1 * z = random noise 
            c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            # c2 * eps = direction pointing to x_t
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            # Eq. (12)
            # Update x_i using the DDIM formula.
            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i