#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 128:00:00
#SBATCH -C a100,ib
#SBATCH -N 5
#SBATCH --gpus=20
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/astrodino-%j.log

module purge
module load python
module load cuda

run_name="astroclip_refactor_final"
config="astroclip/astrodino/config.yaml"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source .local/env.sh

srun python -m astroclip.astrodino.trainer \
    --config-file=$config \
