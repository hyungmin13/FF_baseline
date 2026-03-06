#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --account=2263025
#SBATCH --job-name=RBC
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
#CUDA_VISIBLE_DEVICES=0 python trainer.py -n RBC_fixed_test1 -c config/RBC/RBC_fixed_test1 & CUDA_VISIBLE_DEVICES=1 python trainer.py -n RBC_fixed_test2 -c config/RBC/RBC_fixed_test2 & CUDA_VISIBLE_DEVICES=2 python trainer.py -n RBC_fixed_test3 -c config/RBC/RBC_fixed_test3 & CUDA_VISIBLE_DEVICES=3 python trainer.py -n RBC_fixed_test4 -c config/RBC/RBC_fixed_test4
#CUDA_VISIBLE_DEVICES=0 python trainer.py -n TBL_SOAP_k16_bound -c config/TBL_syn/TBL_SOAP_k16_bound & CUDA_VISIBLE_DEVICES=1 python trainer.py -n TBL_SOAP_k32_bound -c config/TBL_syn/TBL_SOAP_k32_bound & CUDA_VISIBLE_DEVICES=2 python trainer.py -n TBL_SOAP_k64_bound -c config/TBL_syn/TBL_SOAP_k64_bound & CUDA_VISIBLE_DEVICES=3 
CUDA_VISIBLE_DEVICES=0 python trainer.py -n HIT_SOAP_k1_test1 -c config/HIT/HIT_SOAP_k1_test1 & CUDA_VISIBLE_DEVICES=1 python trainer.py -n HIT_SOAP_k2_test1 -c config/HIT/HIT_SOAP_k2_test1 & CUDA_VISIBLE_DEVICES=2 python trainer.py -n HIT_SOAP_k4_test1 -c config/HIT/HIT_SOAP_k4_test1 & CUDA_VISIBLE_DEVICES=3 python trainer.py -n HIT_SOAP_k8_test1 -c config/HIT/HIT_SOAP_k8_test1
