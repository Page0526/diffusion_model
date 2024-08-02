import torch
from torch import nn


# Sample a batch from training data
def sample_batch(dataloader, device):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    data = images.to(device)
    return torch.nn.functional.interpolate(data, 32)