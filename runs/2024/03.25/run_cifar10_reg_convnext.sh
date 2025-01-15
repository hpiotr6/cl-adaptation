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
result_date="03.25"

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn cifar10_convnext_reg \
    data.datasets=["cifar10_fixed"] \
    training.vcreg.reg_layers="classifier$" \
    misc.results_path=results/2024/"${result_date}"-convnext \
    data.num_workers=$num_workers \
    misc.exp_name="reg_fc" &

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn cifar10_convnext_reg \
    data.datasets=["cifar10_fixed"] \
    misc.results_path=results/2024/"${result_date}"-convnext \
    data.num_workers=$num_workers \
    misc.exp_name="reg_deeper" &

wait
