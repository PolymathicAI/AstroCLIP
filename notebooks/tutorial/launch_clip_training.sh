#!/bin/bash -l
#SBATCH -J logitatt  # create a short name for your job
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH -C a100-80gb
#SBATCH -o logit_clip.out

module purge
module load cuda
module load python
source /mnt/home/lparker/python_envs/dino/bin/activate

$JOB_NAME = "logitatt"
$EMBEDDING_DIR = "/mnt/home/lparker/ceph/bad_embeddings"

python clip_dino_training_attblock.py --embedding_dir $EMBEDDING_DIR --wandb_name $JOB_NAME --epochs 100
