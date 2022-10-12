#!/bin/bash
#SBATCH --account=wlr@v100
#SBATCH --job-name=celeba      # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
##SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=10:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=logs/%x_%A_%a.out        # nom du fichier de sortie
#SBATCH --error=logs/%x_%A_%a.out         # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0               # 20 travaux en tout mais 5 travaux max dans la file


# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load pytorch-gpu/py3/1.10.1

conda activate mmvae

wandb offline

# echo des commandes lancees
set -x

python3 src/bivae/dcca/trainings/main_celeba.py --num_epochs 100 --outdim_size 40
python3 src/bivae/main.py --experiment celeba --model jnf_celeba --obj jmvae_nf --latent-dim 168 --warmup 50 --epochs 100 --beta-prior 1 --llik_scaling 1 --data-path '/gpfsscratch/rech/wlr/uhw48em/mmvae_data/'


