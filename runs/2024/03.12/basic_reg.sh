#!/bin/bash
set -e

num_workers=$1
var_weight=$2
cov_weight=$3
smooth_cov=$4
reg_layers=$5
network=$6
result_date=$7
lr=$8
# exp_name=$8

exp_name="${reg_layers}:var_${var_weight}:cov_${cov_weight}:lr_${lr}"

python3.10 -m src.main_incremental \
    --num-workers "${num_workers}" \
    --var_weight "${var_weight}" \
    --cov_weight "${cov_weight}" \
    --exp-name "${exp_name}" \
    --scale True \
    --varcov_reg \
    --reg_layers "${reg_layers}" \
    --smooth_cov "${smooth_cov}" \
    --eval-on-train \
    --approach finetuning \
    --network "${network}" \
    --datasets cifar100_fixed \
    --lr "${lr}" \
    --log wandb disk \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --batch-size 128 \
    --nepochs 100 \
    --num-tasks 5 \
    --results-path results/2024/"${result_date}" \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models
