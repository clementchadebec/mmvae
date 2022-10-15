#!/bin/bash
#SBATCH --job-name=wandb_syncing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/wlr/uhw48em/mmvae/logs/wandb_logs/%x-%j.out           # output file name
#SBATCH --account=wlr@v100

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.10.1

conda activate pythae_tens

# echo des commandes lancees
set -x

#WORKING_DIR=/gpfswork/rech/wlr/uhw48em/benchmark_VAE/examples/scripts
#cd $WORKING_DIR

#export WANDB_DIR=$SCRATCH/
#export WANDB_CACHE_DIR=$SCRATCH/
wandb sync wandb/offline-run-20221014_153159-2sbf2wab





#wandb sync --sync-all
