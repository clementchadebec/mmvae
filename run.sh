#!/bin/bash

##### Circles-squares #####

# JMVAE original formulation
#python3 src/main.py --experiment jmvae --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 0.1 --decrease-beta-kl 1.1 --warmup 15 --fix-decoders False --fix-jencoder False --no-nf --epochs 30 --data-path ../data/circles_squares/ --no-recon True --eval-mode --use-pretrain '../experiments/jmvae/2022-06-28/2022-06-28T13:36:56.9830721rhs9oz0/'


# TELBO original formulation
#python3 src/main.py --experiment jmvae --model jnf_circles_squares --obj telbo_nf --latent-dim 2  --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True --no-nf --skip-warmup True --eval-mode --use-pretrain '../experiments/jmvae/2022-06-28/2022-06-28T17:25:01.03903846svjh2d/'


# JMVAE-NF while fixing the decoders-encoder and decreasing beta_kl and adding the reconstruction term
#python3 src/main.py --experiment jmvae --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 1 --decrease-beta-kl 1 --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True

# JMVAE-NF no recon term
#python3 src/main.py --experiment jmvae_fid --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta-kl 1 --decrease-beta-kl 1 --warmup 15 --epochs 30 --data-path ../data/circles_squares/ --skip-warmup True --no-recon True

# MMVAE
#python3 src/main.py --experiment jmvae_fid --model circles_squares --obj dreg --dist normal --latent-dim 2 --epochs 30 --K 10 --data-path ../data/circles_squares/ --no-nf

##### MNIST-SVHN ######

# JMVAE-NF
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --beta-kl 0.5 --decrease-beta-kl 0.85

# JMVAE
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-recon True --no-nf --skip-warmup True

# JMVAE-recon
#python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-nf --skip-warmup True --decrease-beta-kl 0.85

# JMVAE-NF no recon term
python3 src/main.py --experiment clean_mnist_svhn --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --warmup 30 --epochs 50 --beta-prior 1 --no-recon True


#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 1 --warmup 15 --epochs 30 --beta-rec 1 --fix-decoders --no-nf
#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 5 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 0.5 --warmup 30 --epochs 60 --beta-rec 1 --fix-decoders --no-nf
#python3 src/main.py --model jnf_mnist_fashion --obj telbo_nf --latent-dim 5 --data-path ../data/unbalanced/  --beta-prior 1 --warmup 15 --epochs 30 --skip-warmup True  --no-analytics


#python3 src/main.py --model mnist_svhn --obj dreg --latent-dim 20 --epochs 45 --no-nf --K 10 --dist laplace
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 0.1 --warmup 40 --epochs 60 --skip-warmup True --experiment clean_mnist_svhn --no-analytics --decrease-beta-kl 0.85
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 1 --warmup 15 --epochs 45 --fix-decoders --skip-warmup --no-nf




#python3 src/main.py --model circles_squares --obj dreg --latent-dim 4 --epochs 15 --K 5 --data-path ../data/circles_squares/ --dist laplace