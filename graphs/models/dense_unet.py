import torch.nn.functional as F
from torch import nn

from .unet import ConvBlock, UpBlock


class UNet(nn.Module):
    '''
    U-net implementation.

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels_seg : int
        number of output classes for the segmentation task.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, in_channels, out_channels_seg=5, dim=3, depth=5, initial_channels=32, channels_list= None):

        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.res_list_seg = nn.ModuleList()
        self.res_list_reg = nn.ModuleList()
        for i in range(self.depth):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.down_path.append(ConvBlock(prev_channels, current_channels))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.up_path.append(UpBlock(prev_channels+current_channels, current_channels))
            prev_channels = current_channels
            self.res_list_seg.append(nn.Conv3d(channels_list[i+1], out_channels_seg, kernel_size=1))
            self.res_list_reg.append(nn.Conv3d(channels_list[i + 1], dim, kernel_size=1))

        self.res_list_seg.append(nn.Conv3d(channels_list[0], out_channels_seg, kernel_size=1))
        self.res_list_reg.append(nn.Conv3d(channels_list[0], dim, kernel_size=1))

    def forward(self, x):
        blocks = []
        out_seg = []
        out_reg = []

        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                                  recompute_scale_factor=False)

        for i, (up, res_seg, res_reg) in enumerate(zip(self.up_path, self.res_list_seg, self.res_list_reg)):

            if i == 0:
                out_seg.append(res_seg(x))
                out_reg.append(res_reg(x))

                x = up(x, blocks[-i - 1])
            else:
                out_seg.append(res_seg(x))
                out_reg.append(res_reg(x))
                x = up(x, blocks[-i - 1])

        out_seg.append(self.res_list_seg[-1](x))
        out_reg.append(self.res_list_reg[-1](x))

        return out_seg, out_reg