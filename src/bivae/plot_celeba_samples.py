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
from models.samplers import GaussianMixtureSampler
from utils import Logger, Timer, unpack_data, update_dict_list, get_mean_std, print_mean_std
from torchvision.utils import make_grid, save_image

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda'
n_samples = 50

wandb.init(project = 'plot_celeba_samples' , entity="asenellart") 


"""A script to plot samples from the different models and saving all the attributes """

models_to_evaluate = [ 'jmvae_nf_dcca/celeba','mmvae/celeba','mvae/celeba', 'jmvae/celeba']
model_dicts = []

# load args from disk if pretrained model path is given
for model_name in models_to_evaluate:
    print(model_name)
    day_path = max(glob.glob(os.path.join('../experiments/' + model_name, '*/')), key=os.path.getmtime)
    model_path = max(glob.glob(os.path.join(day_path, '*/')), key=os.path.getmtime)
    with open(model_path + 'args.json', 'r') as fcc_file:
        # Load the args
        wandb.init(project = 'plot_celeba_samples' , entity="asenellart") 

        args = argparse.Namespace()
        args.__dict__.update(json.load(fcc_file))
        # Get the model class
        modelC = getattr(models, 'VAE_{}'.format(args.model))
        # Create instance and load the state dict
        model_i = modelC(args).to(device)
        print('Loading model {} from {}'.format(model_i.modelName, model_path))
        model_i.load_state_dict(torch.load(model_path + '/model.pt'))
        # Save everything
        model_dicts.append(dict(args = args,path=model_path, model=model_i))




# Save everything in '../experiments/compare_celeba/'

# set up run path

runPath = Path('../experiments/compare_celeba')
runPath.mkdir(parents=True, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)


train_loader, test_loader, val_loader = model_dicts[0]['model'].getDataLoaders(n_samples, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")



def compare_samples():
    """Compute all metrics on the entire test dataset"""

    # Take the first batch of the test_dataloader as conditional samples
    dataT = next(iter(test_loader))
    data = unpack_data(dataT, device=device)
    classes = dataT[0][1], dataT[1][1]
    # Sample from the conditional distribution for each model 
    samples = []
    for m in model_dicts:
        model = m['model']
        model.eval()
        s = model._sample_from_conditional(data, n=10) # s[i][j] is a list with n tensors (n_batch, c, w, h)
        s = torch.stack(s[1][0]).permute(1,0,2,3,4) # n_batch x n_samples x c x w x h
        samples.append(s)
    samples = torch.stack(samples).permute(1,0,2,3,4,5)

    for i, t in enumerate(samples):
        # t is of size n_models, 10, c, w, h
        kwargs = dict(nrow = 10)
        save_image(t.reshape(len(t)*10, *t.shape[2:]), str(runPath) + f'/samples_{i}.png',**kwargs)
        # get the attributes 
        att = np.argwhere(classes[0][i] ==1)[0].numpy()
        

        l = np.array(train_loader.dataset.attr_names)[att]
        
        np.savetxt(str(runPath) + f'/attributes_{i}.txt', l,fmt="%s")

    return 



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        compare_samples()