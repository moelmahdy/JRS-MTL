import os

from utils.config import process_config_gen
from utils.generate_jobs import submit_job

setting = dict()
setting['cluster_manager'] = 'Slurm'
setting['NumberOfGPU'] = 1
setting['cluster_MemPerCPU'] = 7500
setting['cluster_NumberOfCPU'] = 7            # Number of CPU per job
setting['cluster_NodeList'] = 'res-hpc-lkeb03' # ['res-hpc-gpu01','res-hpc-gpu02','res-hpc-lkeb03',---,'res-hpc-lkeb07']


if 'lkeb' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'LKEBgpu'
    setting['cluster_Partition'] = 'LKEBgpu'
elif 'gpu' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'gpu'
    setting['cluster_Partition'] = 'gpu'

experiments_dict = {}
experiments_dict['segmentation_a'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}

experiments_dict['segmentation_b'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If_Sm', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}


experiments_dict['segmentation_c'] ={'model_name':'Seg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Seg',
                                     'input':'If_Im_Sm', "task_ids": ['seg'], 'num_featurmaps': [23, 45, 91], 'num_classes':5}


experiments_dict['registration_a'] ={'model_name':'Reg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Reg',
                                     'input':'If_Im', 'task_ids': ['reg'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}

experiments_dict['registration_b'] ={'model_name':'Reg', 'task':'Single-Task', 'agent':'stlAgent', 'network':'Reg',
                                     'input':'If_Im_Sm', 'task_ids': ['reg'], 'num_featurmaps': [23, 45, 91], 'num_classes':3}


exp = experiments_dict['segmentation_a']
exp['is_debug'] = False
exp['mode'] = 'train' #['train', 'inference', 'eval']

base_json_script = '/exports/lkeb-hpc/mseelmahdy/JRS-MTL/configs/base_args.json'
script_address = '/exports/lkeb-hpc/mseelmahdy/JRS-MTL/main.py'
root_log_path = os.path.join('/exports/lkeb-hpc/mseelmahdy/JRS-MTL/experiments', exp['task'])

if exp['task'] == 'Single-Task':
    exp_name = f"{exp['model_name']}_input_{exp['input']}"
elif exp['task'] == 'Multi-Task':
    exp_name = f'{exp["model_name"]}_inSeg_{exp["input_seg"]}_inReg_{exp["input_reg"]}_lSeg_{exp["loss_seg"]}_lReg_{exp["loss_reg"]}_{exp["weight"]}'
if exp['is_debug']:
    exp_name = f'{exp_name}_debug'


config = process_config_gen(base_json_script, exp_name, exp)
json_script = os.path.join(config.log_dir)
submit_job(exp_name, script_address, setting=setting, root_log_path=root_log_path, mode=exp['mode'], json_script=json_script)