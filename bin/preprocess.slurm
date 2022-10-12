#!/bin/bash
#SBATCH --job-name=preprocess_celeba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/wlr/uhw48em/mmvae/bin/logs/%x-%j.out           # output file name
#SBATCH --account=wlr@v100

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.10.1

conda activate mmvae

# echo des commandes lancees
set -x

python download_celeba.py

