#!/bin/bash


# Run all the experiments and the validation for mnist-svhn

# JMVAE_NF_DCCA
python3 src/bivae/dcca/trainings/main_celeba.py --config-path src/configs_experiments/celeba/jmvae_nf_dcca.json
python3 src/bivae/main.py --config-path src/configs_experiments/celeba/jmvae_nf_dcca.json

# JMVAE
python3 src/bivae/main.py --config-path src/configs_experiments/celeba/jmvae.json

# JMVAE_NF
python3 src/bivae/main.py --config-path src/configs_experiments/celeba/jmvae_nf.json

# MMVAE
python3 src/bivae/main.py --config-path src/configs_experiments/celeba/mmvae.json

# MVAE
python3 src/bivae/main.py --config-path src/configs_experiments/celeba/mvae.json


# # Run all the validations and compute likelihoods

# # Before that, train the classifiers
python3 src/bivae/analysis/classifiers/CelebA_classifier.py 

# # JMVAE_NF_DCCA ((checked))
python3 src/bivae/validate.py --model jmvae_nf_dcca/celeba 

# # JMVAE
# python3 src/bivae/validate.py --model jmvae/celeba

# # JMVAE_NF
# python3 src/bivae/validate.py --model jmvae_nf/celeba

# # MMVAE
# python3 src/bivae/validate.py --model mmvae/celeba

# # MVAE (checked)
# python3 src/bivae/validate.py --model mvae/celeba

# # Compute all the likelihoods

# # JMVAE_NF_DCCA (checked)
# python3 src/bivae/compute_likelihoods.py --model jmvae_nf_dcca/celeba

# # JMVAE
# python3 src/bivae/compute_likelihoods.py --model jmvae/celeba

# # JMVAE_NF
# python3 src/bivae/compute_likelihoods.py --model jmvae_nf/celeba

# # MMVAE (checked)
# python3 src/bivae/compute_likelihoods.py --model mmvae/celeba

# # MVAE (checked)
# python3 src/bivae/compute_likelihoods.py --model mvae/celeba