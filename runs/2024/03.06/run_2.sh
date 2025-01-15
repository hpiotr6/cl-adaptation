#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 7:00:00
#SBATCH --ntasks 2
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=8

export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
eval "$(conda shell.bash hook)"
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

num_workers=8
result_date="03.06"

./runs/2024/03.06/basic_reg.sh "${num_workers}" 0.64 12.8 1000 BasicBlocks resnet34_skips "${result_date}" &
./runs/2024/03.06/basic_reg.sh "${num_workers}" 12.8 12.8 1 BasicBlocks resnet34_skips "${result_date}" &

wait
