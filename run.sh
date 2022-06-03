#!/bin/bash

##### Circles-squares #####

#python3 src/main.py --model jnf_circles_squares --obj jmvae_nf --latent-dim 2 --beta 1 --data-path ../data/circles_squares/  --beta-prior 1 --warmup 5 --epochs 10


python3 src/main.py --model jnf_mnist_fashion --obj jmvae_nf --latent-dim 10 --beta 1 --data-path ../data/unbalanced/  --beta-prior 1 --warmup 5 --epochs 10
