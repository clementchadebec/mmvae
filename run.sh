#!/bin/bash

##### Circles-squares #####

#python3 src/main.py --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta 1 --data-path ../data/circles_squares/  --beta-prior 1 --warmup 15 --epochs 30 --beta-rec 1 --fix-decoders
python3 src/main.py --model jnf_mnist_fashion --obj jmvae_nf --latent-dim 5 --beta 1 --data-path ../data/unbalanced/  --beta-prior 1 --warmup 10 --epochs 20 --beta-rec 1 --fix-decoders
