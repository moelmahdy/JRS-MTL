import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sys.path.append('/exports/lkeb-hpc/mseelmahdy/NiftyNet')

from niftynet.utilities.util_common import ParserNamespace
from niftynet.io.image_reader import ImageReader
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from utils.balanced_sampler import BalancedSampler as bs
from utils.pad import PadLayer

class DatasetNiftySampler(Dataset):
    """
    A simple adapter
    converting NiftyNet sampler's output into PyTorch Dataset properties
    """
    def __init__(self, sampler):
        super(DatasetNiftySampler, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        data = self.sampler(idx=index)

        # Transpose to PyTorch format
        fimage = np.transpose(data['fixed_image'], (0, 5, 1, 2, 3, 4))
        flabel = np.transpose(data['fixed_segmentation'], (0, 5, 1, 2, 3, 4))
        mimage = np.transpose(data['moving_image'], (0, 5, 1, 2, 3, 4))
        mlabel = np.transpose(data['moving_segmentation'], (0, 5, 1, 2, 3, 4))
        fimage = torch.from_numpy(fimage).float()
        flabel = torch.from_numpy(flabel).float()
        mimage = torch.from_numpy(mimage).float()
        mlabel = torch.from_numpy(mlabel).float()

        return fimage, flabel, mimage, mlabel

    def __len__(self):
        return len(self.sampler.reader.output_list)

def set_dataParam(args, config):
    # Defining the data parameters
    data_param = dict()

    # , loader = 'simpleitk'
    if 'fixed_image' in args.input_list:
        data_param['fixed_image'] = ParserNamespace(csv_file=config.csv_fixed_image,
                                                    spatial_window_size=args.patch_size,
                                                    pixdim=args.voxel_dim, interp_order=3)
    if 'fixed_segmentation' in args.input_list:
        data_param['fixed_segmentation'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_label,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)
    if 'fixed_bladder' in args.input_list:
        data_param['fixed_bladder'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_bladder,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)
    if 'fixed_rectum' in args.input_list:
        data_param['fixed_rectum'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_rectum,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)
    if 'fixed_sv' in args.input_list:
        data_param['fixed_sv'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_sv,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)
    if 'fixed_ln' in args.input_list:
        data_param['fixed_ln'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_ln,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)
    if 'fixed_gtv' in args.input_list:
        data_param['fixed_gtv'] = ParserNamespace(csv_file=config.csv_fixed_segmentation_gtv,
                                                           spatial_window_size=args.patch_size,
                                                           pixdim=args.voxel_dim, interp_order=0)


    if 'moving_image' in args.input_list:
        data_param['moving_image'] = ParserNamespace(csv_file=config.csv_moving_image,
                                                     spatial_window_size=args.patch_size,
                                                     pixdim=args.voxel_dim, interp_order=3)
    if 'moving_segmentation' in args.input_list:
        data_param['moving_segmentation'] = ParserNamespace(csv_file=config.csv_moving_segmentation_label,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)
    if 'moving_bladder' in args.input_list:
        data_param['moving_bladder'] = ParserNamespace(csv_file=config.csv_moving_segmentation_bladder,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)
    if 'moving_rectum' in args.input_list:
        data_param['moving_segmentation'] = ParserNamespace(csv_file=config.csv_moving_segmentation_rectum,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)
    if 'moving_sv' in args.input_list:
        data_param['moving_sv'] = ParserNamespace(csv_file=config.csv_moving_segmentation_sv,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)
    if 'moving_ln' in args.input_list:
        data_param['moving_ln'] = ParserNamespace(csv_file=config.csv_moving_segmentation_ln,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)
    if 'moving_gtv' in args.input_list:
        data_param['moving_gtv'] = ParserNamespace(csv_file=config.csv_moving_segmentation_gtv,
                                                            spatial_window_size=args.patch_size,
                                                            pixdim=args.voxel_dim, interp_order=0)


    if 'sampler' in args.input_list:
        data_param['sampler'] = ParserNamespace(csv_file=config.csv_sampler,
                                                spatial_window_size=args.patch_size,
                                                pixdim=args.voxel_dim,
                                                interp_order=0)


    return data_param


def get_reader(args, data_param, image_sets_partitioner, phase):
    # preprocessing layers
    pad_layer = PadLayer(image_name='', border=args.padding_size, mode='minimum')
    # Using Nifty Reader
    if phase == 'training':
        image_reader = ImageReader().initialise(data_param, file_list=image_sets_partitioner.get_file_list(TRAIN))
        image_reader.add_preprocessing_layers([pad_layer])
    elif phase == 'validation':
        image_reader = ImageReader().initialise(data_param, file_list=image_sets_partitioner.get_file_list(VALID))
        if args.mode == 'train':
            image_reader.add_preprocessing_layers([pad_layer])
    elif phase == 'inference':
        image_reader = ImageReader().initialise(data_param, file_list=image_sets_partitioner.get_file_list(INFER))
    else:
        raise Exception('Invalid phase choice: {}'.format({'phase': args.split_set}))

    return image_reader

def get_sampler(args, image_reader, phase):

    if phase in ('training', 'validation'):
        window_sizes = {}
        for i in range(len(args.input_list)):
            window_sizes[args.input_list[i]] = args.patch_size

        sampler = bs(image_reader, window_sizes=args.patch_size, queue_length =40, windows_per_image=args.windows_per_volume)

    elif phase == 'inference':
        sampler = GridSampler(image_reader,
                              window_sizes=args.patch_size,
                              window_border=args.padding_size,
                              batch_size=1)
    else:
        raise Exception('Invalid phase choice: {}'.format(
            {'phase': args.split_set}))

    return sampler

def get_datasets(args, config):
    # Dictionary with data parameters for NiftyNet Reader
    data_param = set_dataParam(args, config)
    image_sets_partitioner = ImageSetsPartitioner().initialise(data_param=data_param,
                                                               data_split_file=config.csv_split_file,
                                                               new_partition=False)


    readers = {x: get_reader(args, data_param, image_sets_partitioner, x)
               for x in args.split_set}
    samplers = {x: get_sampler(args, readers[x], x)
                for x in args.split_set}

    dsets = {x: DatasetNiftySampler(sampler=samplers[x])
             for x in args.split_set}

    return dsets


