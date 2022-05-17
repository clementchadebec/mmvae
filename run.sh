#!/bin/bash

##### Circles-discs #####

betas=(100)
#for i in ${!betas[@]};
#do python3 src/main.py --model j_circles_discs --obj jmvae --latent-dim 2 --beta ${betas[i]} --data-path ../data/unbalanced/circles_and_discs/ --epochs 5;
# done
#
#
for i in ${!betas[@]};
#do python3 src/main.py --model j_circles_discs --obj jmvae --latent-dim 2 --beta ${betas[i]} --data-path ../data/unbalanced/circles_and_discs/ --epochs 30 --warmup 10
do python3 src/main.py --model circles_discs --obj vaevae_kl --latent-dim 2 --beta ${betas[i]} --data-path ../data/circles_squares/ --beta-prior 0.01
done

##### Mnist-Fashion #####
#betas=(1000, 10000)
#for i in ${!betas[@]};
#do python3 src/main.py --model mnist_fashion --obj vaevae_kl --beta ${betas[i]} --epochs 5
# done

#python3 src/main.py --model j_mnist_fashion --obj telbo --epochs 5;