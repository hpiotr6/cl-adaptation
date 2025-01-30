#!/bin/bash
#SBATCH -A plgdynamic2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 1-0:0
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=16
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=jupyter_notebook

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#!/bin/bash
module load Miniconda3/4.9.2
module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0
conda init bash
conda activate /net/tscratch/people/plghpiotr/.conda/mgr_env

# Set up the Jupyter Notebook environment
export XDG_RUNTIME_DIR=""
export JUPYTER_RUNTIME_DIR=$HOME/.jupyter/runtime

# Start Jupyter Notebook
/net/tscratch/people/plghpiotr/.conda/mgr_env/bin/python3 -m jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
