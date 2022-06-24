#!/bin/bash

##### Circles-squares #####

#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 1 --warmup 15 --epochs 30 --beta-rec 1 --fix-decoders --no-nf
#python3 src/main.py --experiment empty_full --model jnf_circles_squares --obj jmvae_nf --latent-dim 5 --beta 1 --data-path ../data/empty_full_bk/  --beta-prior 0.5 --warmup 30 --epochs 60 --beta-rec 1 --fix-decoders --no-nf
python3 src/main.py --model jnf_mnist_fashion --obj jmvae_nf --latent-dim 5 --data-path ../data/unbalanced/  --beta-prior 1 --warmup 15 --epochs 30 --fix-decoders --skip-warmup


#python3 src/main.py --model mnist_svhn --obj dreg --latent-dim 20 --epochs 45 --no-nf --K 10 --dist laplace
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 0.1 --warmup 40 --epochs 100 --fix-decoders --skip-warmup
#python3 src/main.py --model jnf_mnist_svhn --obj jmvae_nf --latent-dim 20 --beta-prior 1 --warmup 15 --epochs 45 --fix-decoders --skip-warmup --no-nf




#python3 src/main.py --model circles_squares --obj dreg --latent-dim 4 --epochs 15 --K 5 --data-path ../data/circles_squares/ --dist laplace