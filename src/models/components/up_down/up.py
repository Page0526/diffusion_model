import torch
from torch import nn


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, C: int) -> None:
        """
        :param C (int): number of input and output channels
        """
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = nn.functional.interpolate(x, size=None, scale_factor=2, mode='nearest')

        x = self.conv(x)
#         print(f"Upsample-x.shape:{x.shape}")
        assert x.shape == (B, C, H * 2, W * 2)
        return x
    
if __name__ == "__main__":
    # test UpSample
    x = torch.randn(2, 1, 32, 32)
    up = UpSample(C=1)
    out = up(x)

    print('***** UpSample *****')
    print('Input:', x.shape)
    print('Output:', out.shape)