#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
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
result_date="03.26"

# Resnet34 ze skipami, gdzie wylaczono ostatnie relu na potrzeby analizy znakow.
# 1. Badanie ewc, ktore wczesniej sie crashowalo
# 2. Badanie znakow z regu;aryzacja deeper i tylko na przedostatniej warstwie

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn no_last_relu \
    training/approach=ewc \
    data.datasets=["cifar10_fixed"] \
    misc.results_path=results/2024/"${result_date}" \
    data.num_workers=$num_workers &

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn no_last_relu_reg \
    data.datasets=["cifar100_fixed"] \
    misc.results_path=results/2024/"${result_date}" \
    data.num_workers=$num_workers \
    training.vcreg.reg_layers=".*after_relu|fc$" \
    misc.exp_name="reg_deeper" &

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn no_last_relu_reg \
    data.datasets=["cifar100_fixed"] \
    misc.results_path=results/2024/"${result_date}" \
    data.num_workers=$num_workers \
    misc.exp_name="reg_fc" &
wait
