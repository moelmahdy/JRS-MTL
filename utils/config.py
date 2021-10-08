import json
import logging
import os
import random
import sys
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint

import tensorflow as tf
import torch
from easydict import EasyDict

from utils.misc import create_dirs


def setup_logging(log_dir, config):
    log_file_format = "[%(levelname)s] - %(asctime)s: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    exp_file_handler = RotatingFileHandler('{}{}.log'.format(log_dir, config.mode), maxBytes=10**6, backupCount=5)
    stdout_handler = logging.StreamHandler(sys.stdout)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(stdout_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(json_file):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """

    print = logging.getLogger().info
    config, _ = get_config_from_json(json_file)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    # get working device
    config.device = str(get_device())
    config.apex_availabel = apex_check()

    if config.deterministic:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        tf.random.set_random_seed(config.seed)
        # np.random.seed(config.seed)
    if config.reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if (config.mode == 'inference') or (config.mode == 'eval'):
        config.split_set =['inference', 'validation']
    elif config.mode == 'train':
        config.split_set =['training', 'validation']

    # create some important directories to be used for that experiment.
    config.model_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "model/")
    config.output_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "output/")
    config.log_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "logs/")
    config.tensorboard_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "tensorboard/")

    create_dirs([config.checkpoint_dir, config.prediction_dir, config.log_dir, config.tensorboard_dir])

    if (config.add_movingSegmentation) and ("crossStitch" not in config.exp_name):
        config.in_channels +=1
    elif (config.add_movingSegmentation) and ("crossStitch" in config.exp_name):
        config.in_channels_seg += 1
        config.in_channels_reg += 1
    else:
        logging.getLogger().error("Number of channels of the input is not clear !")


    #save version of the configuration file to the experiment log path
    with open(os.path.join(config.log_dir, 'args_'+config.mode+'.json'), 'w') as fp:
        json.dump(config, fp, indent=2)

    # setup logging in the project
    setup_logging(config.log_dir, config)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    print(" THE Configuration of your experiment ..")
    pprint(config)

    return config

def process_config_gen(json_file, exp_name, exp_dict):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """

    print = logging.getLogger().info
    config, _ = get_config_from_json(json_file)
    config.exp_name = exp_name
    config.agent = exp_dict['agent']
    config.network = exp_dict['network']
    config.mode = exp_dict['mode']
    config.task = exp_dict['task']
    config.is_debug = exp_dict['is_debug']
    config.num_featurmaps = exp_dict['num_featurmaps']
    config.task_ids = exp_dict['task_ids']
    if "num_classes" in exp_dict:
        config.num_classes = exp_dict["num_classes"]
    if "weight" in exp_dict:
        if exp_dict['weight'] is None:
            config.weight = 'None'
        else:
            config.weight = exp_dict['weight']

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    # get working device
    config.device = str(get_device())
    config.apex_available = apex_check()

    if config.deterministic:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        tf.random.set_random_seed(config.seed)
        # np.random.seed(config.seed)
    if config.reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if (config.mode == 'inference') or (config.mode == 'eval'):
        config.split_set =['inference', 'validation']
    elif config.mode == 'train':
        config.split_set =['training', 'validation']

    if 'Seg' in exp_dict['model_name']:
        config.label_name = ["Segmentation.mha"]
        config.excel_name = ["Evaluation-Seg.xlsx"]
    elif 'Reg' in exp_dict['model_name'] or 'JRS' in exp_dict['model_name']:
        config.label_name = ["ResampledSegmentation.mha"]
        config.excel_name = ["Evaluation-Reg.xlsx"]
    else:
        config.label_name = ["Segmentation.mha", "ResampledSegmentation.mha"]
        config.excel_name = ["Evaluation-Seg.xlsx", "Evaluation-Reg.xlsx"]

    # create some important directories to be used for that experiment.
    config.model_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "model/")
    config.output_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "output/")
    config.log_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "logs/")
    config.tensorboard_dir = os.path.join(config.root_exp_path, config.task, config.exp_name, "tensorboard/")

    create_dirs([config.model_dir, config.output_dir, config.log_dir, config.tensorboard_dir])

    if "input_seg" in exp_dict:
        if exp_dict['input_seg'] is not None:
            config.input_segmentation = exp_dict['input_seg'].split('_')
        else:
            config.input_segmentation = 'None'

    if "input_reg" in exp_dict:
        if exp_dict['input_reg'] is not None:
            config.input_registration = exp_dict['input_reg'].split('_')
        else:
            config.input_registration = 'None'

    if "loss_seg" in exp_dict:
        if exp_dict['loss_seg'] is not None:
            config.loss_segmentation = exp_dict['loss_seg'].split('_')
        else:
            config.loss_segmentation = 'None'

    if "loss_reg" in exp_dict:
        if exp_dict['loss_reg'] is not None:
            config.loss_registration = exp_dict['loss_reg'].split('_')
        else:
            config.loss_registration = exp_dict['loss_reg']

    if "input" in exp_dict:
        config.input = exp_dict['input'].split('_')

    #save version of the configuration file to the experiment log path
    with open(os.path.join(config.log_dir, 'args_'+config.mode+'.json'), 'w') as fp:
        json.dump(config, fp, indent=2)

    # setup logging in the project
    # setup_logging(config.log_dir, config)
    #
    # logging.getLogger().info("Hi, This is root.")
    # logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    # logging.getLogger().info("The pipeline of the project will begin now.")

    print(" THE Configuration of your experiment ..")
    pprint(config)

    return config

def get_device():

    from utils.misc import print_cuda_statistics

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger().info('-' * 10 + "Operation will be on GPU-CUDA" + '-' * 10)
        print_cuda_statistics()
    else:
        device = torch.device("cpu")
        logging.getLogger().info('-' * 10 + "Operation will be on CPU" + '-' * 10)

    return device

def apex_check():
    try:
        from apex import amp

        logging.getLogger().info('NVIDIA-Apex is available for training')
        APEX_AVAILABLE = True
    except:
        logging.getLogger().info('NVIDIA-Apex not found')
        APEX_AVAILABLE = False

    return APEX_AVAILABLE