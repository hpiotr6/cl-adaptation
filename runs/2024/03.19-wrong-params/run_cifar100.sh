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
result_date="03.19"

./runs/2024/03.19/noreg.sh "${num_workers}" no_last_relu "${result_date}" lwf cifar100_fixed &
./runs/2024/03.19/noreg.sh "${num_workers}" no_last_relu "${result_date}" ewc cifar100_fixed &
./runs/2024/03.19/noreg_exemplars.sh "${num_workers}" no_last_relu "${result_date}" finetuning cifar100_fixed &

wait
