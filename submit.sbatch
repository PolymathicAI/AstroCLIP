#!/bin/bash -l

#SBATCH -p gpu
##SBATCH -C "h100"
#SBATCH -J "astroclip-job"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=200G
#SBATCH --output=logs/astroclip-%j.log
#SBATCH --time=48:00:00

module load gcc
module load nccl

export num_workers=$(expr $SLURM_JOB_CPUS_PER_NODE - 1)
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

# some debugging logs
export WANDB_START_METHOD=thread
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1.

# Load running environment
source /mnt/home/lparker/python_envs/toto/bin/activate

srun $@ \
    --data.num_workers=8 \
    --trainer.num_nodes=${SLURM_NNODES} \
    --trainer.devices=${SLURM_GPUS_PER_NODE} \
    --trainer.strategy='ddp_find_unused_parameters_true'
