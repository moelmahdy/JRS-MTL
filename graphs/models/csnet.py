import torch
import torch.nn as nn

from utils.SpatialTransformer import SpatialTransformer
from . import cs_unet


class CSNet(nn.Module):
    '''
    Groupwise implicit template CNN registration method.
    Parameters
    ----------
    dim : int
        Dimension of input image.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    '''

    def __init__(self, in_channels, dim, classes= 5, depth=3, initial_channels=64, channels_list = None):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_Classes = classes

        self.unet = cs_unet.UNet(in_channels=in_channels, out_channels_seg= classes, dim=dim,
                                 depth=depth, initial_channels=initial_channels, channels_list=channels_list)
        self.spatial_transform = SpatialTransformer(self.dim)


    def forward(self, fixed_image, moving_image, moving_label=None):
        '''
        Parameters
        ----------
        fixed_image, moving_image: (n, 1, d, h, w)
            Fixed and moving image to be registered
        moving_label : optional, (n, 1, d, h, w)
            Moving label
        Returns
        -------
        warped_moving_image : (n, 1, d, h, w)
            Warped moving image.
        disp : (n, 3, d, h, w)
            Flow field from fixed image to moving image.
        '''

        input_image = torch.cat((fixed_image, moving_image), dim=1)  # (n, 2, d, h, w)
        if moving_label is not None:
            input_image = torch.cat((input_image, moving_label), dim=1)  # (n, 3, d, h, w)

        logits_list, disp_list = self.unet(input_image)  # (n, 6, d, h, w), (n, 3, d, h, w)

        res = {'logits_low': logits_list[0], 'logits_mid': logits_list[1], 'logits_high': logits_list[2],
               'dvf_low': disp_list[0], 'dvf_mid': disp_list[1], 'dvf_high': disp_list[2]}

        return res