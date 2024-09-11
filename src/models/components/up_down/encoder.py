from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# NOTE: in the init_block already import all kinds of block & attention
from src.models.components.up_down.down import DownSample
from src.models.components.attention.attention import AttentionBlock
from src.models.components.blocks.residual_block import ResidualBlock

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                #  block: str='Residual',
                 n_layer_blocks: int=1,
                 drop_rate: float = 0.0,
                 channel_multipliers: List[int] = [1, 2, 4],
                #  attention: str='Attention'
                 ):
        """_summary_

            Args:
                in_channels (int): is the number of channels in the input.
                z_channels (int, optional): is the number of channels in the embedding space. Defaults to 3.
                base_channels (int, optional): is the number of channels in the first convolution layer. Defaults to 64.
                block (str, optional): is the block of block in each layers of encoder. Defaults to "Residual".
                n_layer_blocks (int, optional): is the number of resnet layers at each resolution. Defaults to 1.
                drop_rate (float, optional): parameter of dropout layer. Defaults to 0..
                channel_multipliers (List[int], optional): the multiplicative factors for number of channels for each level. Defaults to [1, 2, 4].
                attention (str, optional): type of attentions for each level. Defaults to "Attention".
        """ 
        super.__init__()
        # Number of levels DownSample
        levels = len(channel_multipliers)
        # Number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]
        channels = base_channels

        # Block to DownSample
        Block = ResidualBlock

        # Attention layer
        Attention = AttentionBlock

        # input conv
        self.encoder_input = nn.Conv2d(in_channels=in_channels,
                                       out_channels=channels,
                                       kernel_size=3,
                                       padding=1)
        # List of top-level blocks
        self.encoder = nn.ModuleList()

        for i in range(levels):
            # add blocks & DownSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks):
                blocks.append(
                    Block(in_channels=channels,
                          out_channels=channels_list[i],
                          drop_rate=drop_rate)
                )    

                channels = channels_list[i]

                down=nn.Module()
                down.block = blocks

                # down sampling at the end of each top level except last one
                if i != levels - 1:
                    down.downSample = DownSample(channels=channels)

                else:
                    down.downSample = nn.Identity()

                self.encoder.append(down)

            # mid block with attention
        self.mid = nn.Sequential(
            Block(in_channels=channels, drop_rate=drop_rate),
            Attention(channels=channels),
            Block(in_channels=channels, drop_rate=drop_rate))

        # output encoder
        self.encoder_output = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=z_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`

        Returns:
            Tensor: _description_
        """
        
        # input convolution
        z = self.encoder_input(x)

        # Top-level blocks
        for encoder in self.encoder:
            # Blocks
            for block in encoder.blocks:
                z = block(z)
            # Down-sampling
            z = encoder.downSample(z)

        # mid block with attention
        z = self.mid(z)

        # Map image space to mean-var in z space
        z = self.encoder_output(z)

        return z