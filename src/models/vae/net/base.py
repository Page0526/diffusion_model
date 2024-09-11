from typing import Dict, Tuple

import torch
from torch import nn
from torch import Tensor
from abc import abstractmethod

class BaseVAE(nn.Module):
    def __init__(self)->None:
        super.__init__()


    def encode(self, img: Tensor)-> Tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    def decode(self, z: Tensor)-> Tensor:
        raise NotImplementedError
    

    @torch.no_grad()
    def sample(self, n_samples: int, device: str=torch.device('cuda')) -> Tensor:
        '''
        sample from latent space & return corresponding image space map.
        '''
        z = torch.randn(n_samples, self.latent_dims[0], self.latent_dims[1], self.latent_dims[2], device=device)

        return self.decode(z)
    
    @abstractmethod
    def forward(self, img: Tensor)->Tuple[Tensor, Dict[str, Tensor]]:
        pass

    @abstractmethod
    def loss_function(self, img: Tensor, recons_img: Tensor, **kwargs) -> Tensor:
        pass