#!/bin/bash
#
#SBATCH --job-name=example-gpu    # Job name for tracking
#SBATCH --partition=falcon,gecko  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6        # Number of CPU threads used by your job
#SBATCH --gres=gpu:1              # Number of GPUs to use 
#SBATCH --time=2-00:00:00         # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=BEGIN # Events to send email on, remove if you don't want this
#SBATCH --output=out # Standard out from your job
#SBATCH --error=err  # Standard error from your job

## Initialisation ##

source /etc/profile.d/modules.sh

module purge
module load CUDA
module load python-ml
cd ~/yolo/
source ~/yolo/venv/bin/activate
nvidia-smi -i $CUDA_VISIBLE_DEVICES
#export YOLO_VERBOSE=False
python3 YOLO.py

