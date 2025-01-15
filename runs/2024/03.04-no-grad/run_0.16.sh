#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 5:00:00
#SBATCH --ntasks 3
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5

export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

./runs/2024/03.04/basic_reg.sh 5 0.16 1.28 &
./runs/2024/03.04/basic_reg.sh 5 0.16 0.32 &
./runs/2024/03.04/basic_reg.sh 5 0.16 0.08 &
wait
