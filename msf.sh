#!/bin/bash

# Prepare the dataset

# python3 bin/make_trimodal.py # To launch from the mmvae directory

# Run all the experiments and the validation for mnist-svhn

# JMVAE_NF_DCCA (checked)
# python3 src/bivae/dcca/trainings/main_mnist_svhn_fashion.py # 50s par epoch x 100 epochs
# python3 src/bivae/main.py --config-path src/configs_experiments/msf/jmvae_nf_dcca.json # 1 min par epoch x 150 epochs 

# JMVAE (checked)
python3 src/bivae/main.py --config-path src/configs_experiments/msf/jmvae.json # 1 min par epoch x 150 epochs

# # JMVAE_NF (checked)
# python3 src/bivae/main.py --config-path src/configs_experiments/msf/jmvae_nf.json # 1 min par epoch x 150 epochs
# # MVAE (checked)
python3 src/bivae/main.py --config-path src/configs_experiments/msf/mvae.json # 1 min par epoch x 150 epochs

# # MMVAE (checked)
python3 src/bivae/main.py --config-path src/configs_experiments/msf/mmvae.json # 1 min 15 par epoch x 150 epochs




# # # Run all the validations and compute likelihoods

# # # Before that, train the classifiers

# python3 src/bivae/analysis/classifiers/classifier_mnist.py --mnist-type numbers
# python3 src/bivae/analysis/classifiers/classifier_SVHN.py 
# python3 src/bivae/analysis/classifiers/classifier_mnist.py --mnist-type fashion



# # JMVAE_NF_DCCA (checked) (max 30 min)
# python3 src/bivae/validate.py --model jmvae_nf_dcca/msf 

# # JMVAE (max 30 min)
# python3 src/bivae/validate.py --model jmvae/msf

# # JMVAE_NF (max 30 min)
# python3 src/bivae/validate.py --model jmvae_nf/msf

# # MMVAE (checked) (max 30 min)
# python3 src/bivae/validate.py --model mmvae/msf

# # MVAE (checked) (max 30 min)
# python3 src/bivae/validate.py --model mvae/msf

# # Compute all the likelihoods

# # JMVAE_NF_DCCA (checked) (time estimate ~ 2h)
# python3 src/bivae/compute_likelihoods.py --model jmvae_nf_dcca/msf

# # JMVAE (time estimate ~ 2h)
# python3 src/bivae/compute_likelihoods.py --model jmvae/msf

# # JMVAE_NF (time estimate ~ 2h)
# python3 src/bivae/compute_likelihoods.py --model jmvae_nf/msf

# # MMVAE (checked) (time estimate ~ 2h)
# python3 src/bivae/compute_likelihoods.py --model mmvae/msf

# # MVAE (checked) (time estimate ~ 2h)
# python3 src/bivae/compute_likelihoods.py --model mvae/msf