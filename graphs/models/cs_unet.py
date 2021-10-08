import torch
import torch.nn.functional as F
from torch import nn

from .unet import ConvBlock, UpBlock


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class UNet(nn.Module):
    '''
    U-net implementation with modifications.
        1. Works for input of 2D or 3D
        2. Change batch normalization to instance normalization

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels_seg : int
        number of output channels.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, in_channels, out_channels_seg, dim=3, depth=3, initial_channels=32, channels_list= None):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.depth = depth
        prev_channels = in_channels
        self.down_path_seg = nn.ModuleList()
        self.down_path_reg = nn.ModuleList()
        self.cs_unit_encoder = []
        self.cs_unit_decoder = []
        self.res_list_seg = nn.ModuleList()
        self.res_list_reg = nn.ModuleList()

        for i in range(self.depth):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]

            self.down_path_seg.append(ConvBlock(prev_channels, current_channels))
            self.down_path_reg.append(ConvBlock(prev_channels, current_channels))
            prev_channels = current_channels
            if i < self.depth-1:
                # define cross-stitch units
                self.cs_unit_encoder.append(nn.Parameter(0.5*torch.ones(prev_channels, 2, 2).cuda(), requires_grad=True))
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
            # define cross-stitch units
            self.cs_unit_decoder.append(nn.Parameter(0.5 * torch.ones(prev_channels, 2, 2).cuda(), requires_grad=True))

        self.res_list_seg.append(nn.Conv3d(channels_list[0], out_channels_seg, kernel_size=1))
        self.res_list_reg.append(nn.Conv3d(channels_list[0], dim, kernel_size=1))

        self.cs_unit_encoder = torch.nn.ParameterList(self.cs_unit_encoder)
        self.cs_unit_decoder = torch.nn.ParameterList(self.cs_unit_decoder)


    def apply_cross_stitch(self, a, b, alpha):

        shape = a.shape
        newshape = [shape[0], shape[1],  shape[2] * shape[3]]  # [bs][n_f][x][y] ==> [bs][n_f][x*y]

        a_flat = a.view(newshape)  # [bs][n_f][x*y]
        b_flat = b.view(newshape)  # [bs][n_f][x*y]

        a_flat = torch.unsqueeze(a_flat, 2)  # [bs][n_f][1][x*y]
        b_flat = torch.unsqueeze(b_flat, 2)  # [bs][n_f][1][x*y]
        a_concat_b = torch.cat([a_flat, b_flat], dim=2)  # [bs][n_f][2][x*y]

        alphas_tiled = torch.unsqueeze(alpha, 0).repeat([shape[0], 1, 1, 1])  # [bs][n_f][2][2]

        out = torch.matmul(alphas_tiled, a_concat_b)  # [bs][n_f][2][2] * [bs][n_f][2][x*y] ==> [bs][n_f][2][x*y]
        out = out.permute(2, 0, 1, 3)  # [2][bs][n_f][x*y]

        out_a = out[0, :, :, :]  # [bs][n_f][x*y]
        out_b = out[1, :, :, :]  # [bs][n_f][x*y]

        out_a = out_a.view(shape)  # [bs][n_f][x][y]
        out_b = out_b.view(shape)  # [bs][n_f][x][y]

        return out_a, out_b

    def forward(self, x):

        blocks_seg = []
        blocks_reg = []
        out_seg = []
        out_reg = []
        x_seg = x.clone()
        x_reg = x.clone()

        for i, (down_seg, down_reg) in enumerate(zip(self.down_path_seg, self.down_path_reg)):

            x_seg = down_seg(x_seg)
            x_reg = down_reg(x_reg)

            if i < self.depth - 1:

                blocks_seg.append(x_seg)
                blocks_reg.append(x_reg)

                x_seg = F.interpolate(x_seg, scale_factor=0.5, mode='trilinear', align_corners=True,
                                      recompute_scale_factor=False)
                x_reg = F.interpolate(x_reg, scale_factor=0.5, mode='trilinear', align_corners=True,
                                      recompute_scale_factor=False)

                x_seg, x_reg = self.apply_cross_stitch(x_seg, x_reg, self.cs_unit_encoder[i])

        for i, (up_seg, up_reg, res_seg, res_reg) in enumerate(zip(self.up_path_seg, self.up_path_reg,
                                                                   self.res_list_seg, self.res_list_reg)):
            if i == 0:
                out_seg.append(res_seg(x))
                out_reg.append(res_reg(x))
                x_seg_before = up_seg(x_seg, blocks_seg[-i - 1])
                x_reg_before = up_reg(x_reg, blocks_reg[-i - 1])
                x_seg, x_reg = self.apply_cross_stitch(x_seg_before, x_reg_before, self.cs_unit_decoder[i])
            else:
                out_seg.append(res_seg(x_seg))
                out_reg.append(res_reg(x_reg))
                x_seg_before = up_seg(x_seg, blocks_seg[-i - 1])
                x_reg_before = up_reg(x_reg, blocks_reg[-i - 1])
                x_seg, x_reg = self.apply_cross_stitch(x_seg_before, x_reg_before, self.cs_unit_decoder[i])

        out_seg.append(self.res_list_seg[-1](x_seg))
        out_reg.append(self.res_list_reg[-1](x_reg))

        return out_seg, out_reg


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dim, normalization, LeakyReLU_slope=0.2):
#         super().__init__()
#         block = []
#         if dim == 2:
#             block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             if normalization:
#                 block.append(nn.InstanceNorm2d(out_channels))
#             block.append(nn.LeakyReLU(LeakyReLU_slope))
#         elif dim == 3:
#             block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
#             if normalization:
#                 block.append(nn.InstanceNorm3d(out_channels))
#             block.append(nn.LeakyReLU(LeakyReLU_slope))
#         else:
#             raise (f'dim should be 2 or 3, got {dim}')
#         self.block = nn.Sequential(*block)
#
#     def forward(self, x):
#         out = self.block(x)
#         return out
#
#
# class UpBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dim, normalization):
#         super().__init__()
#         self.dim = dim
#         if dim == 2:
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         elif dim == 3:
#             self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#         self.conv_block = ConvBlock(in_channels, out_channels, dim, normalization)
#
#     def forward(self, x, skip):
#         x_up = F.interpolate(x, skip.shape[2:], mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
#         x_up_conv = self.conv(x_up)
#         out = torch.cat([x_up_conv, skip], 1)
#         out = self.conv_block(out)
#         return out