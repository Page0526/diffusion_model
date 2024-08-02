import torch
from torch import nn


# Sampling
class Downsample(nn.Module):

    def __init__(self, C):
        """
        :param C (int): number of input and output channels
        """
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1) # input_shape, output_shape, kernel_size, stride, padding

    def forward(self, x):
        B, C, H, W = x.shape # batch, channels, height, weight
        x = self.conv(x)
#         print(f"Downsample-x.shape:{x.shape}")
        assert x.shape == (B, C, H // 2, W // 2) # // = divide with integer result
        return x