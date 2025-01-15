#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 8:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1

#!/bin/bash
export WANDB_API_KEY="47f70459596d37c22af356c72cbe0e8467c66e45"
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

python3.10 -m src.main_incremental \
    --approach mas \
    --seed 2 \
    --datasets cifar10 \
    --lr 0.1 \
    --exp-name Bartek_mas_seed_2 \
    --log disk \
    --num-workers 14 \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --batch-size 128 \
    --eval-on-train \
    --nepochs 100 \
    --num-tasks 5 \
    --results-path results/11.29_Bartek \
    --use-test-as-val \
    --scheduler-milestones 30 60 80 \
    --save-models
