#!/bin/bash

########################################################################################################################
############################################### CIRCLES- SQUARES #######################################################
########################################################################################################################

# JMVAE original formulation
#python3 src/main.py --experiment jmvae --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 0.1 --decrease-beta-kl 1.1 --warmup 15 --fix-decoders False --fix-jencoder False --no-nf --epochs 30 --data-path ../data/circles_squares/ --no-recon True --eval-mode --use-pretrain '../experiments/jmvae/2022-06-28/2022-06-28T13:36:56.9830721rhs9oz0/'


# TELBO original formulation
#python3 src/main.py --experiment jmvae --model jnf_circles_squares --obj telbo_nf --latent-dim 2  --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True --no-nf --skip-warmup True --eval-mode --use-pretrain '../experiments/jmvae/2022-06-28/2022-06-28T17:25:01.03903846svjh2d/'


# JMVAE-NF while fixing the decoders-encoder and decreasing beta_kl and adding the reconstruction term
#python3 src/main.py --experiment jmvae_fid --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 1 --decrease-beta-kl 1 --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True --eval-mode --use-pretrain '../experiments/jmvae/2022-06-28/2022-06-28T16:20:36.890838aby5rthj/'

# JMVAE-NF no recon term
#python3 src/main.py --experiment jmvae_fid --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 1 --decrease-beta-kl 1 --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True --no-recon True  #--use-pretrain '../experiments/jmvae/2022-06-30/2022-06-30T13:39:29.893742hog5i_9f/'

# MMVAE
#python3 src/main.py --experiment jmvae_fid --model circles_squares --obj dreg --dist normal --latent-dim 2 --epochs 30 --K 10 --data-path ../data/circles_squares/ --no-nf

########################################################################################################################
###########################offline###################### MNIST-FASHION ########################################################
########################################################################################################################

# JMVAE
#python3 src/main.py --experiment jnf_mnist_fashion_fid --model jnf_mnist_fashion --obj jmvae_nf --no-nf --latent-dim 5 --data-path ../data/unbalanced/ --warmup 15 --epochs 30 --no-recon True --eval-mode --use-pretrain '../experiments/jnf_mnist_fashion_fid/2022-07-13/2022-07-13T14:43:32.776808szbp5eef/'

# MMVAE
#python3 src/main.py --experiment jnf_mnist_fashion_fid --model mnist_fashion --obj dreg --dist normal --latent-dim 5 --epochs 30 --K 10 --data-path ../data/unbalanced/ --no-nf

# JMVAE-NF
#python3 src/main.py --experiment jnf_mnist_fashion_fid --model jnf_mnist_fashion --obj jmvae_nf --latent-dim 5 --data-path ../data/unbalanced/ --warmup 15 --epochs 30 --loss 'l1'

########################################################################################################################
################################################## MNIST-SVHN ##########################################################
########################################################################################################################

# JMVAE original formulation
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --epochs 50 --beta-prior 1 --beta-kl 0.1 --decrease-beta-kl 1.1 --warmup 30 --skip-warmup True --fix-jencoder False --fix-decoders False --no-nf --no-recon True

# JMVAE-NF
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --beta-kl 0.5 --decrease-beta-kl 0.85

# JMVAE
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-recon True --no-nf --skip-warmup True

# JMVAE-recon
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-nf --skip-warmup True --decrease-beta-kl 0.85

# JMVAE-NF no recon term
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-recon True --skip-warmup True

# JMVAE-NF-DCCA with reconstruction term
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn_dcca --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --skip-warmup True

# JMVAE-NF-DCCA with no reconstruction term
#python3 src/bivae/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn_dcca --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --skip-warmup True --no-recon True



########################################################################################################################
########################################## MNIST-CONTOUR ###############################################################
########################################################################################################################

python3 src/bivae/main.py --experiment contour --model jnf_mnist_contour --obj jmvae_nf --latent-dim 15 --warmup 15 --epochs 30 --beta-prior 1




########################################################################################################################
########################################### CIRCLES-SQUARES INVERSE ####################################################
########################################################################################################################

# JMVAE-NF
#python3 src/main.py --experiment inverse --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 1 --decrease-beta-kl 1 --warmup 15 --epochs 30 --data-path ../data/circles_squares_inverse/




#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 1 --warmup 15 --epochs 30 --beta-rec 1 --fix-decoders --no-nf
#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 5 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 0.5 --warmup 30 --epochs 60 --beta-rec 1 --fix-decoders --no-nf
#python3 src/main.py --model jnf_mnist_fashion --obj telbo_nf --latent-dim 5 --data-path ../data/unbalanced/  --beta-prior 1 --warmup 15 --epochs 30 --skip-warmup True  --no-analytics


#python3 src/main.py --model mnist_svhn --obj dreg --latent-dim 20 --epochs 45 --no-nf --K 10 --dist laplace
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 0.1 --warmup 40 --epochs 60 --skip-warmup True --experiment clean_mnist_svhn --no-analytics --decrease-beta-kl 0.85
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 1 --warmup 15 --epochs 45 --fix-decoders --skip-warmup --no-nf




#python3 src/main.py --model circles_squares --obj dreg --latent-dim 4 --epochs 15 --K 5 --data-path ../data/circles_squares/ --dist laplace