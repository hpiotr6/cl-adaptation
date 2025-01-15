#!/bin/bash
set -e
python3.10 -m src.main_incremental \
    --exp-name fin_reg_rev \
    --varcov_reg \
    --var_weight 12.8 \
    --cov_weight 6.4 \
    --approach finetuning \
    --network resnet34_skips \
    --datasets cifar100_fixed \
    --lr 0.1 \
    --log wandb disk \
    --num-workers 8 \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --batch-size 128 \
    --nepochs 100 \
    --num-tasks 5 \
    --results-path results/2024/02.28 \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models \
    --tags reg_on_grads reg_before_fc mean_dim_0
