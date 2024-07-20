import torch
from torch import nn

# Sampling
class DownSample(nn.Module):
    """
    ### Down-sampling layer
    """

    def __init__(self, C: int) -> None:
        """
        :param C (int): number of input and output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1) # input_shape, output_shape, kernel_size, stride, padding

    def forward(self, x):
        B, C, H, W = x.shape # batch, channels, height, weight
        x = self.conv(x)
#         print(f"Downsample-x.shape:{x.shape}")
        assert x.shape == (B, C, H // 2, W // 2) # // = divide with integer result
        return x
    
if __name__ == "__main__":
    # test DownSample
    x = torch.randn(2, 1, 32, 32)
    up = DownSample(C=1)
    out = up(x)

    print('***** DownSample *****')
    print('Input:', x.shape)
    print('Output:', out.shape)