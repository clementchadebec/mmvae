#!/bin/bash

########################################################################################################################
############################################### CIRCLES- SQUARES #######################################################
########################################################################################################################

# MMVAE
#python3 src/validate.py --use-pretrain '../experiments/jmvae_fid/2022-07-04/2022-07-04T13:26:51.053979zpb_41pr/'

########################################################################################################################
################################################# MNIST-FASHION ########################################################
########################################################################################################################

# JMVAE
#python3 src/validate.py --use-pretrain '../experiments/jnf_mnist_fashion_fid/2022-07-13/2022-07-13T14:43:32.776808szbp5eef/'

# JMVAE-NF-no-decrease
#python3 src/validate.py --use-pretrain '../experiments/jnf_mnist_fashion_fid/2022-06-27/2022-06-27T12:55:38.6541363kwsc2fu/'

# MMVAE
#python3 src/validate.py --use-pretrain '../experiments/jnf_mnist_fashion_fid/2022-07-19/2022-07-19T14:00:34.606762zyv9i0ca/'

# MMVAE normal
#python3 src/validate.py --use-pretrain '../experiments/jnf_mnist_fashion_fid/2022-08-17/2022-08-17T14:45:16.110386ht_hugh6/'
# TELBO

# Compare PRD plots
#python3 src/analysis/plot_prds_data.py -l ../experiments/jnf_mnist_fashion_fid/2022-06-27/2022-06-27T12:55:38.6541363kwsc2fu/ ../experiments/jnf_mnist_fashion_fid/2022-07-19/2022-07-19T14:00:34.606762zyv9i0ca/ -n jmvae_nf mmvae

# Compare unimodal PRD plots
#python3 src/analysis/plot_uniprds_data.py -l ../experiments/jnf_mnist_fashion_fid/2022-06-27/2022-06-27T12:55:38.6541363kwsc2fu/ ../experiments/jnf_mnist_fashion_fid/2022-07-19/2022-07-19T14:00:34.606762zyv9i0ca/ -n jmvae_nf mmvae

# Compare Inception embeddings
#python3 src/analysis/plot_inception_embeddings.py -l ../experiments/jnf_mnist_fashion_fid/2022-06-27/2022-06-27T12:55:38.6541363kwsc2fu/ ../experiments/jnf_mnist_fashion_fid/2022-07-19/2022-07-19T14:00:34.606762zyv9i0ca/ ../experiments/jnf_mnist_fashion_fid/2022-07-13/2022-07-13T14:43:32.776808szbp5eef/

########################################################################################################################
################################################## MNIST-SVHN ##########################################################
########################################################################################################################


# JMVAE
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-06-30/2022-06-30T10:59:26.039425jmi11iup/'

# JMVAE-NF
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-06-29/2022-06-29T15:03:18.435049accbhxdz/'

# MMVAE
#python3 src/validate.py --use-pretrain '../experiments/mmvae_mnist_svhn/2022-06-20/2022-06-20T18:39:18.5986649k3o_d2e/'

# JMVAE-NF-DCCA-no-recon
python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-07-06/2022-07-06T14:50:16.455935b4yig7bt/'

# JMVAE-NF-DCCA
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-07-20/2022-07-20T10:22:05.309847keje2c1k/'

# JMVAE-NF-no-recon
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-06-30/2022-06-30T14:12:35.379016xez57359/'