import torch
from torch import nn


# Sampling
class DownSample(nn.Module):

    def __init__(self, channels: int)->None:
        """
        :param channels (int): number of input and output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        return self.conv(x)