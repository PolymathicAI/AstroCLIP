#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 128:00:00
#SBATCH -C a100,ib
#SBATCH -N 5
#SBATCH --gpus=20
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12

module purge
module load python
module load cuda
source /mnt/home/lparker/python_envs/dino/bin/activate

# --- SPECIFY THE FOLLOWING --- 

run_name="astroclip_refactor_final"
config="/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/astrodino/configs/astro_vitl12_simplified_mlp_momentum_snd.yaml"

# ----------------------------- 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/mnt/home/flanusse/.local/lib/python3.10/site-packages:/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2/

srun python -m astrodino.train.train \
    --config-file=$config \
    --output-dir=/mnt/home/lparker/ceph/astroclip_final_refactor \
    --run-name=$run_name \

