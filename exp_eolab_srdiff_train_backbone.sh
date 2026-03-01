#!/bin/bash 
# #SBATCH --job-name=exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:a100m40:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=48:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4_%j.out
# #SBATCH --error logs/exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4_%j.err
# source load_modules.sh

export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/my_data/"
export DATA_DIR
source /home/eouser/flair_venv/bin/activate
which python
cd $HOME/exp_2026/MISR_JOINT_SRDiff_LCC_FLAIR3_HR_INFER
python trainer.py --config_file=./configs/train_main/ --exp_name misr/srdiff_maxvit_ltae_ckpt --reset
