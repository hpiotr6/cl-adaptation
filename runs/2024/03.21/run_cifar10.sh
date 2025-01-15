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
result_date="03.21"

python3.10 -m src.main_incremental \
    --multirun \
    -cp ../runs/2024/"${result_date}" -cn no_last_relu \
    training/approach=lwf,ewc \
    data.datasets=["cifar10_fixed"] \
    misc.results_path=results/2024/"${result_date}" \
    data.num_workers=$num_workers &

python3.10 -m src.main_incremental \
    -cp ../runs/2024/"${result_date}" -cn no_last_relu \
    training/approach=finetuning \
    data.datasets=["cifar10_fixed"] \
    misc.results_path=results/2024/"${result_date}" \
    data.num_workers=$num_workers \
    data.exemplars.num_exemplars_per_class=20 \
    data.exemplars.exemplar_selection=herding &

wait
