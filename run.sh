#!/bin/bash

##### Circles-discs #####

betas=(1)
#for i in ${!betas[@]};
#do python3 src/main.py --model j_circles_discs --obj jmvae --latent-dim 2 --beta ${betas[i]} --data-path ../data/unbalanced/circles_and_discs/ --epochs 5;
# done
#
#
for i in ${!betas[@]};
do python3 src/main.py --model jnf_circles_discs --obj jmvae_nf --latent-dim 2 --beta ${betas[i]} --data-path ../data/circles_squares/ --num-hidden-layers 1 --beta-prior 3 --warmup 15 --epochs 30

#do python3 src/main.py --model circles_discs --obj vaevae_kl --latent-dim 2 --beta ${betas[i]} --data-path ../data/circles_squares/ --beta-prior 0.1 --warmup 0 --epochs 50

done

##### Mnist-Fashion #####
#betas=(1000, 10000)
#for i in ${!betas[@]};
#do python3 src/main.py --model j_mnist_fashion --obj jmvae --beta ${betas[i]}
# done

#python3 src/main.py --model j_mnist_fashion --obj telbo --epochs 5;