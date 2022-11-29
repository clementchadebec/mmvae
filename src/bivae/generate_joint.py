import argparse
import datetime
import json
import random
import sys
from pathlib import Path
import os, glob
from tqdm import tqdm

import numpy as np
import torch
import wandb

import models
from bivae.models.samplers import GaussianMixtureSampler
from bivae.utils import Logger, Timer, unpack_data, update_dict_list, get_mean_std, print_mean_std, load_joint_vae
from bivae.analysis.accuracies import compute_joint_accuracy

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--modality', type=str, default='')
info = parser.parse_args()

# Path to the pretrained joint encoder
model_path = '../experiments/joint_encoders/'+ info.modality + '/'


# load args from disk if pretrained model path is given
with open(model_path + 'args.json', 'r') as fcc_file:
    args = argparse.Namespace()
    args.__dict__.update(json.load(fcc_file))

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

args.device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
print(f'Device is {args.device}')
device = torch.device(args.device)

# Create instance of the model
print(args.model)
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)


# Load joint_vae state_dict
print('Loading model {} from {}'.format(model.modelName, model_path))
load_joint_vae(model,model_path)



train_loader, test_loader, val_loader = model.getDataLoaders(2000, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")




# Define a sampler for generating new samples
# model.sampler = None


def generate(n, batchsize = 1000):
    """Compute all metrics on the entire test dataset"""
    model.eval()
    # Compute all train latents
    model.compute_all_train_latents(train_loader)

    # re-fit the sampler before computing metrics
    if model.sampler is not None:
        model.sampler.fit_from_latents(model.train_latents[0])
        
    
    cpter = min(batchsize, n)
    samples = [[], []]
    while cpter <= n:

        # Sample from the joint latent space
        samples_ = model.generate(N=batchsize, runPath='', epoch=0)
        
        # Check the coherence between modalities
        bdata = [s[:1000] for s in samples_ ]
        model.set_classifiers()
        joint_acc = compute_joint_accuracy(model, bdata)
        print('Joint accuracy {}'.format(joint_acc))
        
        for m, l in enumerate(samples):
            l.append(samples_[m])
        
        cpter += min(batchsize, n-cpter)

    # Save the samples to reuse later
    for i,s in enumerate(samples):
        print(torch.cat(s).shape)
        torch.save(torch.cat(s),model_path + '/generated_modality_{}.pt'.format(i) )


if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        for n_components in [20]:
            n_samples = 200000
            model.sampler = GaussianMixtureSampler(n_components=n_components)
            print("Sampling {} samples with n_components = {}".format(n_samples, n_components))
            generate(n=n_samples)
            
        