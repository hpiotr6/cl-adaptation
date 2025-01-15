#!/bin/bash
#SBATCH -A plggenerativepw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 24:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16

#!/bin/bash
module load Miniconda3/4.9.2
source activate /net/tscratch/people/plghpiotr/.conda/mgr_env

python3.10 -m src.tunnel_project.tunnel_analysis \
    --config-name 01.09.yaml \
    +checkpoint_path=results/2024/01.08/cifar10/cifar10_fixed_finetuning_resnet34_skips_reg_deeper_1000
