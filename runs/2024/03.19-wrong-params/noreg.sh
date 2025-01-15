#!/bin/bash
set -e

num_workers=$1
network=$2
result_date=$3
approach=$4
dataset=$5

python3.10 -m src.main_incremental \
    --num-workers "${num_workers}" \
    --scale False \
    --eval-on-train \
    --approach "${approach}" \
    --network "${network}" \
    --datasets "${dataset}" \
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
