import logging
import os
import time

import imageio
import matplotlib.pylab as plt
import numpy as np
import torchvision
from IPython import display


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))

def generate_gif(volume, gif_name='out.gif'):
    #create an image for each interpolation
    images = []
    for i in range(volume.shape[2]):
        images.append(volume[0,0,i, :, :].numpy())
    imageio.mimsave(gif_name, images)

def display_gif(gif_path='out.gif'):
    with open(gif_path,'rb') as f:
        display.Image(data=f.read(), format='png')

def create_dirs(dirs):
    """
        dirs - a list of directories to create if these directories are not found
        :param dirs:
        :return:
        """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)

def generate_grid(volume, slice_num):

    grid_img = torchvision.utils.make_grid(volume[:, :, slice_num, :, :],
                                           nrow=int(np.sqrt(volume.shape[0])),
                                           normalize=True,
                                           padding=10,
                                           pad_value=0)
    plt.imshow(grid_img.permute((1, 2, 0)), cmap='gray')

    return grid_img


