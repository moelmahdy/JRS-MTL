import numpy as np
import torch

'''
    Parse the config file
    ----------
    file : json
        config file
    '''
def parser(file):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    import yaml
    return (dotdict(yaml.load(open(file))))

class StopCriterion(object):
    def __init__(self, stop_std=0.001, query_len=100, num_min_iter=200):
        self.query_len = query_len
        self.stop_std = stop_std
        self.loss_list = []
        self.loss_min = 1.
        self.num_min_iter = num_min_iter

    def add(self, loss):
        self.loss_list.append(loss)
        if loss < self.loss_min:
            self.loss_min = loss
            self.loss_min_i = len(self.loss_list)

    def stop(self):
        # return True if the stop creteria are met
        query_list = self.loss_list[-self.query_len:]
        query_std = np.std(query_list)
        if query_std < self.stop_std and self.loss_list[-1] - self.loss_min < self.stop_std / 3. and len(
                self.loss_list) > self.loss_min_i and len(self.loss_list) > self.num_min_iter:
            return True
        else:
            return False



def resize_image_mlvl(args, image, level):
    return (image[:, :, (args.mlvl_borders[level]):(args.patch_size[0] - args.mlvl_borders[level]),
            (args.mlvl_borders[level]):(args.patch_size[1] - args.mlvl_borders[level]),
            (args.mlvl_borders[level]):(args.patch_size[2] - args.mlvl_borders[level])]) \
            [:, :, ::args.mlvl_strides[level], ::args.mlvl_strides[level], ::args.mlvl_strides[level]]

def clean_data(fimage, flabel, mimage, mlabel, args):
    # format the input images in the right format for pytorch model
    nbatches, wsize, nchannels, x, y, z, _ = fimage.size()
    fimage = fimage.view(nbatches * wsize, nchannels, x, y, z).to(args.device)  # (n, 1, d, w, h)
    flabel = flabel.view(nbatches * wsize, nchannels, x, y, z).to(args.device)
    mimage = mimage.view(nbatches * wsize, nchannels, x, y, z).to(args.device)
    mlabel = mlabel.view(nbatches * wsize, nchannels, x, y, z).to(args.device)

    # normalize image intensity
    fimage[fimage > 1000] = 1000
    fimage[fimage < -1000] = -1000
    fimage = fimage / 1000

    mimage[mimage > 1000] = 1000
    mimage[mimage < -1000] = -1000
    mimage = mimage / 1000

    # resize the images for different resolutions
    fimage = fimage.to(args.device)  # size B*C*D*W,H
    flabel = flabel.to(args.device).float()
    mimage = mimage.to(args.device)  # size B*C*D*W,H
    mlabel = mlabel.to(args.device).float()

    flabel_high = resize_image_mlvl(args, flabel, 0)
    flabel_mid = resize_image_mlvl(args, flabel, 1)
    flabel_low = resize_image_mlvl(args, flabel, 2)

    fimage_high = resize_image_mlvl(args, fimage, 0)
    fimage_mid = resize_image_mlvl(args, fimage, 1)
    fimage_low = resize_image_mlvl(args, fimage, 2)

    mlabel_high = resize_image_mlvl(args, mlabel, 0)
    mlabel_mid = resize_image_mlvl(args, mlabel, 1)
    mlabel_low = resize_image_mlvl(args, mlabel, 2)

    mimage_high = resize_image_mlvl(args, mimage, 0)
    mimage_mid = resize_image_mlvl(args, mimage, 1)
    mimage_low = resize_image_mlvl(args, mimage, 2)

    mlabel_high_hot = torch.eye(args.num_classes_seg)[mlabel_high.squeeze(1).long()]
    mlabel_high_hot = mlabel_high_hot.permute(0, 4, 1, 2, 3).float().to(args.device)

    mlabel_mid_hot = torch.eye(args.num_classes_seg)[mlabel_mid.squeeze(1).long()]
    mlabel_mid_hot = mlabel_mid_hot.permute(0, 4, 1, 2, 3).float().to(args.device)

    mlabel_low_hot = torch.eye(args.num_classes_seg)[mlabel_low.squeeze(1).long()]
    mlabel_low_hot = mlabel_low_hot.permute(0, 4, 1, 2, 3).float().to(args.device)

    data_dict = {'fimage': fimage, 'mimage': mimage, 'mlabel': mlabel,
                 'flabel_high':flabel_high, 'flabel_mid':flabel_mid, 'flabel_low':flabel_low,
                 'fimage_high':fimage_high, 'fimage_mid':fimage_mid, 'fimage_low':fimage_low,
                 'mlabel_high':mlabel_high, 'mlabel_mid':mlabel_mid, 'mlabel_low':mlabel_low,
                 'mimage_high':mimage_high, 'mimage_mid':mimage_mid, 'mimage_low':mimage_low,
                 'mlabel_high_hot':mlabel_high_hot, 'mlabel_mid_hot':mlabel_mid_hot, 'mlabel_low_hot':mlabel_low_hot}

    return data_dict
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

