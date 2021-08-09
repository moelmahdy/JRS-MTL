import torch
import torch.nn as nn
import torch.nn.functional as F

from . import unet


class SegNet(nn.Module):
    '''
    CNN segmentation network.
    Parameters
    ----------

    in_channels : int, optional
        number of channels of the input image. The default is 1
    classes : int, optional
        number of classes of the input label image. The default is 5
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    channels_list: list, optional
        list of number of featur maps at every depth in case of customized number of feature maps.
    '''

    def __init__(self, in_channels=1, classes= 5, depth=5, initial_channels=64, channels_list = None):

        super().__init__()
        self.num_Classes = classes
        self.channels_list = channels_list
        assert len(channels_list) == depth

        self.unet = unet.UNet(in_channels=in_channels, out_channels=classes, depth=depth,
                              initial_channels=initial_channels, channels_list= channels_list)

    def forward(self, input_image):
        '''
        Parameters
        ----------
        input_image : (n, 1, d, h, w)
            The first dimension contains the number of patches.
        Returns
        -------
        logits : (n. classes, d, h, w)
            logits of the images.
        '''

        original_image_shape = input_image.shape[2:]
        logits = torch.squeeze(self.unet(input_image), 0).reshape(self.n,  self.num_Classes, *original_image_shape)  #(n, classes, d, h, w)
        probs = F.softmax(logits, dim=1)
        _, predicted_label = torch.max(probs, dim=1, keepdim=True)

        res = {'logits': logits, 'probs': probs, 'predicted_label': predicted_label}

        return res