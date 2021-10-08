import torch.nn.functional as F
from torch import nn

from .unet import ConvBlock, UpBlock


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class UNet(nn.Module):
    '''
    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels_seg : int
        number of output classes for the segmentation task.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    '''

    def __init__(self, in_channels, out_channels_seg, dim=3, depth=3, initial_channels=32, channels_list= None):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
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

        self.up_path_seg = nn.ModuleList()
        self.up_path_reg = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.up_path_seg.append(UpBlock(prev_channels+current_channels, current_channels))
            self.up_path_reg.append(UpBlock(prev_channels+current_channels, current_channels))
            prev_channels = current_channels
            self.res_list_seg.append(nn.Conv3d(channels_list[i + 1], out_channels_seg, kernel_size=1))
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

        for i, (up_seg, up_reg, res_seg, res_reg) in enumerate(zip(self.up_path_seg, self.up_path_reg, self.res_list_seg, self.res_list_reg)):
            if i == 0:
                out_seg.append(res_seg(x))
                out_reg.append(res_reg(x))
                x_seg = up_seg(x, blocks[-i - 1])
                x_reg = up_reg(x, blocks[-i - 1])

            else:
                out_seg.append(res_seg(x_seg))
                out_reg.append(res_reg(x_reg))
                x_seg = up_seg(x_seg, blocks[-i - 1])
                x_reg = up_reg(x_reg, blocks[-i - 1])

        out_seg.append(self.res_list_seg[-1](x_seg))
        out_reg.append(self.res_list_reg[-1](x_reg))

        return out_seg, out_reg
