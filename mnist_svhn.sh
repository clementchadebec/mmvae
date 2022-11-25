#!/bin/bash


# Run all the experiments and the validation for mnist-svhn

# JMVAE_NF_DCCA
python3 src/bivae/dcca/trainings/main_mnist_svhn.py
python3 src/bivae/main.py --config-path src/configs_experiments/mnist_svhn/jmvae_nf_dcca.json

# JMVAE
# python3 src/bivae/main.py --config-path src/configs_experiments/mnist_svhn/jmvae.json

# # JMVAE_NF
# python3 src/bivae/main.py --config-path src/configs_experiments/mnist_svhn/jmvae_nf.json

# # MMVAE
# python3 src/bivae/main.py --config-path src/configs_experiments/mnist_svhn/mmvae.json

# # MVAE
# python3 src/bivae/main.py --config-path src/configs_experiments/mnist_svhn/mvae.json


# # # Run all the validations and compute likelihoods

# # # Before that, train the classifiers

# python3 src/bivae/analysis/classifiers/classifier_mnist.py --mnist-type numbers
# python3 src/bivae/analysis/classifiers/classifier_SVHN.py 

# # # JMVAE_NF_DCCA
# # python3 src/bivae/validate.py --model jmvae_nf_dcca/mnist_svhn

# # # JMVAE
# # python3 src/bivae/validate.py --model jmvae/mnist_svhn

# # # JMVAE_NF
# # python3 src/bivae/validate.py --model jmvae_nf/mnist_svhn

# # # MMVAE (checked)
# # python3 src/bivae/validate.py --model mmvae/mnist_svhn

# # # MVAE (checked)
# # python3 src/bivae/validate.py --model mvae/mnist_svhn

# # # Compute all the likelihoods

# # # JMVAE_NF_DCCA
# # python3 src/bivae/compute_likelihoods.py --model jmvae_nf_dcca/mnist_svhn

# # # JMVAE
# # python3 src/bivae/compute_likelihoods.py --model jmvae/mnist_svhn

# # # JMVAE_NF
# # python3 src/bivae/compute_likelihoods.py --model jmvae_nf/mnist_svhn

# # # MMVAE
# # python3 src/bivae/compute_likelihoods.py --model mmvae/mnist_svhn

# # MVAE
# python3 src/bivae/compute_likelihoods.py --model mvae/mnist_svhn
