import os


def job_script(setting, job_name=None, script_address=None, job_output_file=None, json_script=None, mode=None):
    text = '#!/bin/bash \n'
    text = text + '#SBATCH --job-name=' + job_name.split('_')[0] + '\n'
    text = text + '#SBATCH --output=' + str(job_output_file) + '\n'
    text = text + '#SBATCH --ntasks=1 \n'
    text = text + '#SBATCH --cpus-per-task=' + str(setting['cluster_NumberOfCPU']) + '\n'
    text = text + '#SBATCH --mem-per-cpu=' + str(setting['cluster_MemPerCPU']) + '\n'
    text = text + '#SBATCH --partition=' + setting['cluster_Partition'] + '\n'

    # text = text + '#SBATCH --mem -0' + '\n'
    if setting['cluster_Partition'] in ['gpu', 'LKEBgpu'] and setting['NumberOfGPU']:
        text = text + '#SBATCH --gres=gpu:' + str(setting['NumberOfGPU']) + ' \n'
    text = text + '#SBATCH --time=0 \n'
    if setting['cluster_NodeList'] is not None:
        text = text + '#SBATCH --nodelist='+setting['cluster_NodeList']+' \n'

    text = text + 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/mseelmahdy/cudnn7.4-for-cuda9.0/cuda/lib64/' '\n'
    text = text + 'source /exports/lkeb-hpc/mseelmahdy/fastMRI-env/bin/activate' '\n'

    text = text + 'echo "on Hostname = $(hostname)"' '\n'
    text = text + 'echo "on GPU      = $CUDA_VISIBLE_DEVICES"' '\n'
    text = text + 'echo' '\n'
    text = text + 'echo "@ $(date)"' '\n'
    text = text + 'echo' '\n'

    text = text + 'python ' + str(script_address) + ' ' + os.path.join(json_script, f'args_{mode}.json')
    text = text + '\n'
    return text

def write_and_submit_job(setting, job_name, script_address, root_log_path=None, mode=None, json_script=None):
    """
    Write a bashscript and submit the bashscript as a job to slurm.
    :param setting:
    :param job_name:
    :param script_address:
    :return:
    """
    job_output_file = os.path.join(json_script, mode + '_job.txt')
    job_script_address = os.path.join(json_script, mode + '_job.sh')

    with open(job_script_address, "w") as string_file:

        string_file.write(job_script(setting, job_name=job_name, script_address=script_address,
                                     job_output_file=job_output_file, json_script=json_script, mode=mode))
        string_file.close()

    submit_cmd = 'sbatch ' + str(job_script_address)
    os.system(submit_cmd)

def submit_job(job_name, script_address, setting=None, root_log_path=None, mode=None, json_script=None):
    # Choosing the preferred setting and backup the whole code and submit the job

    if setting is None:
        # Slurm
        setting = dict()
        setting['cluster_queue'] = 'LKEBgpu'  #['gpu', 'LKEBgpu']
        setting['cluster_manager'] = 'Slurm'  #'Slurm'
        setting['NumberOfGPU'] = 1
        setting['cluster_NodeList'] = 'res-hpc-lkeb03'  #['res-hpc-lkeb01', 'res-hpc-lkeb02', 'res-hpc-lkeb03',
        # 'res-hpc-lkeb04', 'res-hpc-lkeb05', 'res-hpc-gpu02', 'res-hpc-gpu01']
        setting['cluster_MemPerCPU'] = 6200
        setting['cluster_NumberOfCPU'] = 10   #Number of CPU per job
        setting['cluster_Partition'] = 'LKEBgpu'  #['gpu', 'LKEBgpu']

    write_and_submit_job(setting, job_name=job_name, script_address=script_address, root_log_path=root_log_path, mode=mode, json_script= json_script)