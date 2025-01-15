#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 1:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16

#!/bin/bash
export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

python3.10 -m src.main_incremental \
    --approach finetuning \
    --network resnet34_skips \
    --datasets cifar100_fixed \
    --lr 0.1 \
    --exp-name resnet34_skips_reg_lay4 \
    --reg_layers layer4 \
    --varcov_reg \
    --var_weight 1 \
    --cov_weight 1 \
    --log wandb disk \
    --num-workers 16 \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --batch-size 128 \
    --eval-on-train \
    --nepochs 100 \
    --num-tasks 2 \
    --results-path results/2024/01.19 \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models
