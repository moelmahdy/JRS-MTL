import os

from utils.config import process_config_gen

setting = dict()
setting['cluster_manager'] = 'Slurm'
setting['NumberOfGPU'] = 1
setting['cluster_MemPerCPU'] = 7500
setting['cluster_NumberOfCPU'] = 7            # Number of CPU per job
setting['cluster_NodeList'] = 'res-hpc-gpu02' # ['res-hpc-lkeb05', 'res-hpc-lkeb03', 'res-hpc-gpu02', 'res-hpc-gpu01']


if 'lkeb' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'LKEBgpu'
    setting['cluster_Partition'] = 'LKEBgpu'
elif 'gpu' in setting['cluster_NodeList']:
    setting['cluster_queue'] = 'gpu'
    setting['cluster_Partition'] = 'gpu'

experiments_dict = {}
experiments_dict['segmentation_a'] ={'is_debug': False, 'model_name':'Seg', 'mode':'train',
                                   'task':'Single-Task', 'agent':'stlAgent', 'network': 'Seg',
                                   'weight':None, 'input_seg':'If', 'input_reg':None,
                                   'loss_seg':'DSC', 'loss_reg':None, "task_ids": ["seg"],
                                    'num_featurmaps': [23, 45, 91]}

experiments_dict['segmentation_b'] ={'is_debug': False, 'model_name':'Seg_mod', 'mode':'train',
                                   'task':'Single-Task', 'agent':'SegmentationAgent',
                                   'weight':None, 'input_seg':'If_Sm', 'input_reg':None,
                                   'loss_seg':'DSC', 'loss_reg':None, "task_ids": ["seg"],
                                    'num_featurmaps': [23, 45, 91]}


experiments_dict['segmentation_c'] ={'is_debug': False, 'model_name':'Seg_mod', 'mode':'train',
                                   'task':'Single-Task', 'agent':'SegmentationAgent',
                                   'weight':None, 'input_seg':'If_Im_Sm', 'input_reg':None,
                                   'loss_seg':'DSC', 'loss_reg':None, "task_ids": ["seg"],
                                    'num_featurmaps': [23, 45, 91]}


experiments_dict['registration_a'] ={'is_debug': False, 'model_name':'Reg_mod2', 'mode':'train',
                                   'task':'Single-Task', 'agent':'RegistrationAgent',
                                   'weight':None, 'input_seg':None, 'input_reg':'If_Im',
                                   'loss_seg':None, 'loss_reg':'NCC', "task_ids": ["reg"],
                                    'num_featurmaps': [23, 45, 91]}

experiments_dict['registration_b'] ={'is_debug': False, 'model_name':'Reg_mod3', 'mode':'train',
                                   'task':'Single-Task', 'agent':'RegistrationAgent',
                                   'weight':None, 'input_seg':None, 'input_reg':'If_Im_Sm',
                                   'loss_seg':None, 'loss_reg':'NCC', "task_ids": ["reg"],
                                    'num_featurmaps': [23, 45, 91]}

experiments_dict['jrs_a'] ={'is_debug': False, 'model_name':'JRS_mod', 'mode':'train',
                          'task':'Multi-Task', 'agent':'RegistrationAgent',
                          'weight':'equal', 'input_seg':None, 'input_reg':'If_Im',
                          'loss_seg':None, 'loss_reg':'NCC_DSCWarp',
                          "task_ids": ["reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['jrs_b'] ={'is_debug': False, 'model_name':'JRS_mod', 'mode':'train',
                            'task':'Multi-Task', 'agent':'RegistrationAgent',
                            'weight':'equal', 'input_seg':None, 'input_reg':'If_Im_Sm',
                            'loss_seg':None, 'loss_reg':'NCC_DSCWarp',
                            "task_ids": ["reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['dense'] ={'is_debug': False, 'model_name':'Dense_mod', 'mode':'train',
                            'task':'Multi-Task', 'agent':'HardSharingAgent',
                            'weight':'equal', 'input_seg':'If_Im_Sm', 'input_reg':'If_Im_Sm',
                            'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                            "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['split'] ={'is_debug': False, 'model_name':'Split_mod', 'mode':'train',
                            'task':'Multi-Task', 'agent':'SplitSharingAgent',
                            'weight':'equal', 'input_seg':'If_Im_Sm', 'input_reg':'If_Im_Sm',
                            'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                            "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['cross-stitch_a'] ={'is_debug': False, 'model_name':'CS_mod', 'mode':'train', 'task':'Multi-Task',
                                     'agent':'CrossStitchWeighingAgent', 'weight':'equal', 'input_seg':'If',
                                     'input_reg':'If_Im', 'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                                     "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['cross-stitch_b'] ={'is_debug': False, 'model_name':'CS_mod', 'mode':'train', 'task':'Multi-Task',
                                     'agent':'CrossStitchWeighingAgent', 'weight':'equal', 'input_seg':'If',
                                     'input_reg':'If_Im_Sm', 'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                                     "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}

experiments_dict['cross-stitch_c'] ={'is_debug': False, 'model_name':'CS_mod', 'mode':'train', 'task':'Multi-Task',
                                     'agent':'CrossStitchWeighingAgent', 'weight':'equal', 'input_seg':'If_Sm',
                                     'input_reg':'If_Im_Sm', 'loss_seg':'DSC', 'loss_reg':'NCC_DSCWarp',
                                     "task_ids": ["seg", "reg", "seg_reg"], 'num_featurmaps': [16, 32, 64]}


is_debug = True
mode = 'train' #['train', 'inference', 'eval']
weight = 'None'
exp = experiments_dict['segmentation_a']
exp['is_debug'] = is_debug
exp['mode'] = mode
exp['weight'] = weight

base_json_script = '/exports/lkeb-hpc/mseelmahdy/JRS-MTL/configs/base_config.json'
script_address = '/exports/lkeb-hpc/mseelmahdy/JRS-MTL/main.py'
root_log_path = os.path.join('/exports/lkeb-hpc/mseelmahdy/JRS-MTL/experiments', exp['task'])

if is_debug:
    exp_name = f'{exp["model_name"]}_inSeg_{exp["input_seg"]}_inReg_{exp["input_reg"]}_lSeg_{exp["loss_seg"]}_lReg_{exp["loss_reg"]}_{exp["weight"]}_debug'
else:
    exp_name = f'{exp["model_name"]}_inSeg_{exp["input_seg"]}_inReg_{exp["input_reg"]}_lSeg_{exp["loss_seg"]}_lReg_{exp["loss_reg"]}_{exp["weight"]}'

if mode == 'all':
    exp['mode'] = 'train'
    config = process_config_gen(base_json_script, exp_name, exp)
    exp['mode'] = 'inference'
    config = process_config_gen(base_json_script, exp_name, exp)
    exp['mode'] = 'eval'
    config = process_config_gen(base_json_script, exp_name, exp)
else:
    config = process_config_gen(base_json_script, exp_name, exp)

json_script = os.path.join(config.log_dir)
# submit_job(exp_name, script_address, setting=setting, root_log_path=root_log_path, mode=mode, json_script=json_script)