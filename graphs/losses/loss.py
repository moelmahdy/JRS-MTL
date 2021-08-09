import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.backends.cudnn.deterministic = True

class NCC(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between tow images.
    Parameters
    ----------
    dim : int
        Dimension of the input images.
    windows_size : int
        Side length of the square window to calculate the local NCC.
    '''

    def __init__(self, dim, windows_size=7):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-4  # numerical stability constant

        self.windows_size = windows_size

        self.pad = windows_size // 2
        self.window_volume = windows_size ** self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        windows_size : int
            Side length of the square window to calculate the local NCC.

        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient.
        '''
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)

        cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)

        cc = cross / ((I_var * J_var) ** 0.5)

        return 1.0-torch.mean(cc)


def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field.

    Parameters
    ----------
    disp : (n, 3, d, h, w)
        displacement field

    image : (1, 1, d, h, w)
    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])

    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)


    d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
    d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
    d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

    d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
    d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
    d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim=2, keepdims=True))

    return loss


def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class multi_dice_loss(nn.Module):
    '''
        Calculate the multi dice loss.

        Parameters
        ----------
        num_classes : int
            Number of classes
    '''
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

    '''
        Parameters
        ----------
        target : (d, h, w)
            The groundtruth label image.
        logits : (classes, d, h, w)
            logits from the network prediction.
        Returns
        -------
            Return 1 - the mean of the dice for all classes excpet of the background.
    '''

    def forward(self, target, prediction, task='seg'):

        target_one_hot = make_one_hot(target.to(torch.int64), self.num_classes)
        if task == 'reg':
            prediction_one_hot = make_one_hot(prediction.to(torch.int64), self.num_classes)
        elif task == 'seg':
            prediction_one_hot = prediction

        inter = (target_one_hot * prediction_one_hot).sum(axis=[-1, -2])
        union = (target_one_hot + prediction_one_hot).sum(axis=[-1, -2])
        dsc_list_per_image = (2. * inter / (union + 1e-3))[:, 1:]
        dsc_mean = 1 - (dsc_list_per_image.mean(axis=0))
        return  dsc_mean



