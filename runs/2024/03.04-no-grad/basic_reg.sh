#!/bin/bash
set -e

num_workers=$1
var_weight=$2
cov_weight=$3

exp_name="var_${var_weight}:cov_${cov_weight}"

python3.10 -m src.main_incremental \
    --num-workers "${num_workers}" \
    --var_weight "${var_weight}" \
    --cov_weight "${cov_weight}" \
    --exp-name "${exp_name}" \
    --varcov_reg \
    --reg_layers BasicBlocks \
    --smooth_cov 1.0 \
    --eval-on-train \
    --approach finetuning \
    --network resnet34_skips \
    --datasets cifar100_fixed \
    --lr 0.1 \
    --log wandb disk \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --batch-size 128 \
    --nepochs 100 \
    --num-tasks 5 \
    --results-path results/2024/03.04-proper-demean \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models \
    --tags on_basicblocks grid_search proper-demean
