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

# MMVAE-looser
#python3 src/bivae/validate.py --use-pretrain '../experiments/mmvae/mnist_svhn/2022-10-24/2022-10-24T14:49:43.799138pgrmanof/'


# JMVAE-NF-DCCA-no-recon
python3 src/bivae/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-10-03/2022-10-03T16:40:02.881385cdcfscvu/'

# JMVAE-NF-DCCA
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-07-20/2022-07-20T10:22:05.309847keje2c1k/'

# JMVAE-NF-no-recon
#python3 src/validate.py --use-pretrain '../experiments/clean_mnist_svhn/2022-06-30/2022-06-30T14:12:35.379016xez57359/'


########################################################################################################################
################################################### CELEBA #############################################################
########################################################################################################################

# JMVAE-NF-DCCA-50000 no recon
#python3 src/bivae/validate.py --use-pretrain '../experiments/celeba/2022-10-14/2022-10-14T11:31:01.7147672x9xu6d4/'

# MMVAE - 20000
#python3 src/bivae/validate.py --use-pretrain '../experiments/mmvae/celeba/2022-10-25/2022-10-25T14:27:05.5670410huxrv_s/'


########################################################################################################################
############################################ COMPUTE LIKELIHOODS #######################################################
########################################################################################################################

################################################# MNIST-SVHN ###########################################################

# JMVAE-NF-DCCA-no-recon
#python3 src/bivae/compute_likelihoods.py --use-pretrain '../experiments/clean_mnist_svhn/2022-10-03/2022-10-03T16:40:02.881385cdcfscvu/'

# MMVAE
#python3 src/bivae/compute_likelihoods.py --use-pretrain '../experiments/mmvae_mnist_svhn/2022-06-20/2022-06-20T18:39:18.5986649k3o_d2e/' --k 1000

# MMVAE-looser
#python3 src/bivae/compute_likelihoods.py --use-pretrain '../experiments/mmvae/mnist_svhn/2022-10-24/2022-10-24T14:49:43.799138pgrmanof/'


################################################## CELEBA ##############################################################

# JMVAE-NF-DCCA-no-recon
# python3 src/bivae/compute_likelihoods.py --use-pretrain '../experiments/celeba/2022-10-14/2022-10-14T11:31:01.7147672x9xu6d4/' --k 1000

# MMVAE - looser
# python3 src/bivae/compute_likelihoods.py --use-pretrain '../experiments/mmvae/celeba/2022-10-25/2022-10-24T14:49:43.799138pgrmanof/' -k 1000
