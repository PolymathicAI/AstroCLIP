#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -C a100-80gb
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH -t 168:00:00
#SBATCH --output=logs/out-%j.log
#SBATCH -J "embedding"

module purge
module load gcc

$dset_root = "/mnt/ceph/users/polymathic/MultimodalUniverse/legacysurvey/dr10_south_21"
$save_root = "/mnt/ceph/users/polymathic/MultimodalUniverse/astrodino_legacysurvey"

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

# enable logging
export  CUDA_LAUNCH_BLOCKING=1.

source /mnt/home/lparker/python_envs/toto/bin/activate

python launch_embeddings.py --dset_root $dset_root --save_root $save_root --batch_size 512 --num_gpus $SLURM_GPUS_PER_NODE
