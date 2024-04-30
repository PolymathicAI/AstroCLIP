#!/bin/bash -l

#SBATCH -p gpu
##SBATCH -C "h100|a100"
#SBATCH -J "astroclip-spectrum"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=20G
#SBATCH --output=logs/astroclip-%j.log
#SBATCH --time=15:00:00

module load nccl

export num_workers=$(expr $SLURM_JOB_CPUS_PER_NODE - 1)
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

# some debugging logs
export WANDB_START_METHOD=thread
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1.

# Load running environment
source .local/env.sh

srun $@ \
    --data.num_workers=${num_workers} \
    --trainer.num_nodes=${SLURM_NNODES} \
    --trainer.devices=${SLURM_GPUS_PER_NODE} \
    --trainer.strategy='ddp_find_unused_parameters_true'
