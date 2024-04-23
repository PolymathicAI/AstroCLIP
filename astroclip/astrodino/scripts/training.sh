#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH -C a100,ib
#SBATCH -N 4
#SBATCH --gpus=16
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -o radial_pos.out

module purge
module load python
module load cuda
source /mnt/home/lparker/python_envs/dino/bin/activate

# --- SPECIFY THE FOLLOWING --- 

run_name="custom_cp"
config="astrodino/configs/ssl_default_config.yaml"

# ----------------------------- 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/mnt/home/flanusse/.local/lib/python3.10/site-packages:/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2/

srun python -m astrodino.train.train \
    --config-file=$config \
    --output-dir=/mnt/home/lparker/ceph/astrodino/$run_name \
    --run-name=$run_name \

