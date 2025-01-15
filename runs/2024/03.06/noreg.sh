#!/bin/bash
set -e

num_workers=$1
result_date=$2

exp_name="noreg"
python3.10 -m src.main_incremental \
    --num-workers "${num_workers}" \
    --exp-name "${exp_name}" \
    --reg_layers BasicBlocks \
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
    --results-path results/2024/"${result_date}" \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models
