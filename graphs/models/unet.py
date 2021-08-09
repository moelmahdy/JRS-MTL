import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    '''
    U-net implementation.

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels : int
        number of output channels.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    channels_list: list, optional
        list of number of featur maps at every depth in case of customized number of feature maps.
    '''

    def __init__(self, in_channels, out_channels, depth=5, initial_channels=32, channels_list= None):

        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.down_path.append(ConvBlock(prev_channels, current_channels))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.up_path.append(UpBlock(prev_channels, current_channels))
            prev_channels = current_channels


        self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                                  recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, LeakyReLU_slope=0.2):
        super().__init__()
        block = []

        block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        block.append(nn.BatchNorm3d(out_channels))
        block.append(nn.LeakyReLU(LeakyReLU_slope))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x_up = F.interpolate(x, skip.shape[2:], mode='trilinear', align_corners=True)
        x_up_conv = self.conv(x_up)
        out = torch.cat([x_up_conv, skip], 1)
        out = self.conv_block(out)
        return out