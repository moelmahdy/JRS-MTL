import torch
import torch.nn as nn

from utils.SpatialTransformer import SpatialTransformer
from . import unet


class RegNet(nn.Module):
    '''
    Pairwise CNN registration network.
    Parameters
    ----------
    dim : int
        Dimension of input image.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, in_channels=2, dim=3, depth=5, initial_channels=64, channels_list = None):

        super().__init__()


        self.unet = unet.UNet(in_channels=in_channels, out_channels=dim, depth=depth, initial_channels=initial_channels,
                              channels_list = channels_list)
        self.spatial_transform = SpatialTransformer(dim=dim)

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

        original_image_shape = fixed_image.shape[2:]
        input_image = torch.unsqueeze(torch.stack((fixed_image, moving_image), dim=0), 0)  # (n, 2, d, h, w)
        if moving_label is not None:
            input_image = torch.unsqueeze(torch.stack((input_image, moving_label), dim=0), 0)  # (n, 3, d, h, w)

        disp = torch.squeeze(self.unet(input_image), 0).reshape(self.dim, *original_image_shape)  #(n, 3, d, h, w)
        disp = torch.unsqueeze(disp, 0)

        warped_moving_image = self.spatial_transform(moving_image, disp, mode='bilinear').squeeze()  #(n, 1, d, h, w)

        res = {'disp': disp.squeeze(0), 'warped_moving_image': warped_moving_image}
        return res