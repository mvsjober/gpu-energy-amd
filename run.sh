#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=5

set -x
srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 ./gpu-energy --save

sleep 5

srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 ./gpu-energy --diff
