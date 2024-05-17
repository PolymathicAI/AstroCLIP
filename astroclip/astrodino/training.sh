#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -C a100,ib
#SBATCH -N 5
#SBATCH --gpus=20
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/astrodino-%j.log

module purge
module load python
module load cuda
module load gcc

random_number=$(shuf -i 2000-65000 -n 1)
run_name="astroclip_$random_number"
config="astroclip/astrodino/config.yaml"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /mnt/home/lparker/python_envs/toto/bin/activate

srun python -m astroclip.astrodino.trainer \
    --config-file=$config --run-name=$run_name
