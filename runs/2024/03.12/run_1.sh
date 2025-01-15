#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 7:00:00
#SBATCH --ntasks 3
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5

export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
eval "$(conda shell.bash hook)"
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

num_workers=5
result_date="03.12"

./runs/2024/03.12/noreg.sh "${num_workers}" convnext_tiny "${result_date}" 0.1 0.0002 "noreg_classic" &
./runs/2024/03.12/noreg.sh "${num_workers}" convnext_tiny "${result_date}" 0.01 0 "no_decay" &
./runs/2024/03.12/noreg.sh "${num_workers}" convnext_tiny "${result_date}" 0.1 0.05 "more_decay" &

wait
