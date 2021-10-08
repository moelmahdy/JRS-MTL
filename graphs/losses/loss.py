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


class GradientSmoothing(nn.Module):
    def __init__(self, energy_type):
        super(GradientSmoothing, self).__init__()
        self.energy_type = energy_type

    def forward(self, dvf):

        def gradient_dx(fv):
            return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

        def gradient_dy(fv):
            return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

        def gradient_dz(fv):
            return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

        def gradient_txyz(Txyz, fn):
            return torch.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], dim=4)

        def compute_gradient_norm(displacement, flag_l1=False):
            dTdx = gradient_txyz(displacement, gradient_dx)
            dTdy = gradient_txyz(displacement, gradient_dy)
            dTdz = gradient_txyz(displacement, gradient_dz)
            if flag_l1:
                norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
            else:
                norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
            return torch.mean(norms)

        def compute_bending_energy(displacement):
            dTdx = gradient_txyz(displacement, gradient_dx)
            dTdy = gradient_txyz(displacement, gradient_dy)
            dTdz = gradient_txyz(displacement, gradient_dz)
            dTdxx = gradient_txyz(dTdx, gradient_dx)
            dTdyy = gradient_txyz(dTdy, gradient_dy)
            dTdzz = gradient_txyz(dTdz, gradient_dz)
            dTdxy = gradient_txyz(dTdx, gradient_dy)
            dTdyz = gradient_txyz(dTdy, gradient_dz)
            dTdxz = gradient_txyz(dTdx, gradient_dz)
            return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

        if self.energy_type == 'bending':
            energy = compute_bending_energy(dvf)
        elif self.energy_type == 'gradient-l2':
            energy = compute_gradient_norm(dvf)
        elif self.energy_type == 'gradient-l1':
            energy = compute_gradient_norm(dvf, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')

        return energy


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


class Multi_DSC_Loss(nn.Module):
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


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        pass

    def forward(self, true, logits, use_activation=True, num_classes=5):

        dice_loss, dice_list = self.dice_loss(true, logits, num_classes, use_activation)

        return dice_loss, dice_list

    def dice_loss(self, true, logits, num_classes, use_activation, eps=1e-7):
        """Computes the SørensenDice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W, D].
            logits: a tensor of shape [B, C, H, W, D]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the SørensenDice loss.
        """
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float().to(true.device)
        if use_activation:
            probas = F.softmax(logits, dim=1)
        else:
            probas = logits
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_list = (2. * intersection / (cardinality + eps))
        dice_loss = dice_list[1:].mean()
        return (1 - dice_loss), dice_list[1:]


class Homoscedastic(torch.nn.Module):
  '''https://arxiv.homoscedasticorg/abs/1705.07115'''
  def __init__(self, n_tasks, reduction='sum'):
    super(Homoscedastic, self).__init__()
    self.n_tasks = n_tasks
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    coeffs = 1/(stds**2)
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses

def grad_norm(self, lossList):
    # get the sum of weighted losses
    task_losses = torch.stack(lossList)
    weighted_losses = self.weights * task_losses
    total_weighted_loss = weighted_losses.sum()

    self.optimizer.zero_grad()

    # compute and retain gradients
    total_weighted_loss.backward(retain_graph=True)

    # GRADNORM - learn the weights for each tasks gradients

    # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
    self.weights.grad = 0.0 * self.weights.grad

    W = [param for name, param in self.model.named_parameters() if (self.args.shared_layers_key in name)]

    norms = []

    for w_i, L_i in zip(self.weights, task_losses):
        # gradient of L_i(t) w.r.t. W
        gLgW = torch.autograd.grad(L_i, W, retain_graph=True)

        # G^{(i)}_W(t)
        norms.append(torch.norm(w_i * gLgW[0], 2))

    norms = torch.stack(norms)

    # set L(0)
    # if using log(C) init, remove these two lines
    if self.current_epoch == 0:
        self.initial_losses = task_losses.detach()

    # compute the constant term without accumulating gradients
    # as it should stay constant during back-propagation
    with torch.no_grad():
        # loss ratios \curl{L}(t)
        loss_ratios = task_losses / self.initial_losses

        # inverse training rate r(t)
        inverse_train_rates = loss_ratios / loss_ratios.mean()

        constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

    # write out the gradnorm loss L_grad and set the weight gradients

    grad_norm_loss = (norms - constant_term).abs().sum()
    self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]

    # apply gradient descent
    self.optimizer.step()

    # renormalize the gradient weights
    with torch.no_grad():
        normalize_coeff = len(self.weights) / self.weights.sum()
        self.weights.data = self.weights.data * normalize_coeff

    return total_weighted_loss


