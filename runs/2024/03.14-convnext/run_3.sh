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
result_date="03.14-convnext"

./runs/2024/03.14-convnext/noreg.sh "${num_workers}" convnext_tiny "${result_date}" 0.016 &
./runs/2024/03.14-convnext/noreg.sh "${num_workers}" convnext_tiny "${result_date}" 0.0016 &
./runs/2024/03.14-convnext/basic_reg.sh "${num_workers}" 0 0 1 ".*after_skipping|classifier$" convnext_tiny "${result_date}" 0.1 &

wait
