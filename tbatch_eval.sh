#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=00:50:00
#SBATCH --account=2263025
#SBATCH --job-name=TBL
##SBATCH --job-name=test
#SBATCH --output=simulation-%j.out
#SBATCH --error=simulation-%j.err
#SBATCH --mail-user=wpark4@naver.com
#SBATCH --mail-type=ALL
#SBATCH --partition=rome-a100
module load env/easybuild
module load Miniforge/23.1.0-2
conda info --envs
conda init --all
conda init bash
source ~/.bashrc
conda activate /home/hyun_sh/.conda/envs/jax_env
nvidia-smi
# Execute application
export CUDA_VISIBLE_DEVICES=1,2,3,4
#CUDA_VISIBLE_DEVICES=0 python tecplot2.py -f RBC_fixed_k1_test1 -c config/eval_config_RBC2 & CUDA_VISIBLE_DEVICES=1 python tecplot2.py -f RBC_fixed_k1_test2 -c config/eval_config_RBC2 & CUDA_VISIBLE_DEVICES=2 python tecplot2.py -f RBC_fixed_k1_test3 -c config/eval_config_RBC2 & CUDA_VISIBLE_DEVICES=3 python tecplot2.py -f RBC_fixed_k1_test4 -c config/eval_config_RBC2
#CUDA_VISIBLE_DEVICES=0 python tecplot.py -f HIT_adam_k1 -c config/eval_config_HIT & CUDA_VISIBLE_DEVICES=1 python tecplot.py -f TBL_adam_k32 -c config/eval_config_TBL_syn & CUDA_VISIBLE_DEVICES=2 python tecplot.py -f TBL_adam_k64 -c config/eval_config_TBL_syn
#CUDA_VISIBLE_DEVICES=3 python tecplot.py -f TBL_SOAP_k1 -c config/eval_config_TBL_syn
CUDA_VISIBLE_DEVICES=0 python tecplot.py -f HIT_SOAP_k1 -c config/eval_config_HIT & CUDA_VISIBLE_DEVICES=1 python tecplot.py -f HIT_SOAP_k2 -c config/eval_config_HIT & CUDA_VISIBLE_DEVICES=2 python tecplot.py -f HIT_SOAP_k4 -c config/eval_config_HIT & CUDA_VISIBLE_DEVICES=3 python tecplot.py -f HIT_SOAP_k8 -c config/eval_config_HIT
