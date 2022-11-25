import argparse
import datetime
import glob
import json
import os
import random
import sys
from pathlib import Path

import bivae.models
import numpy as np
import torch
from tqdm import tqdm
from bivae.utils import (Logger, Timer, get_mean_std,
                   print_mean_std, unpack_data, update_dict_list)

import wandb

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--k', type=int, default=1000)


# args
info = parser.parse_args()

# load args from disk if pretrained model path is given
# Take the last trained model in that folder
day_path = max(glob.glob(os.path.join('../experiments/' + info.model, '*/')), key=os.path.getmtime)
model_path = max(glob.glob(os.path.join(day_path, '*/')), key=os.path.getmtime)
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

# Define parameters for Wandb logging
experiment_name = args.wandb_experiment if hasattr(args, 'wandb_experiment') else args.model
wandb.init(project = experiment_name , entity="asenellart") 
wandb.config.update(args)
wandb.define_metric('epoch')
wandb.define_metric('*', step_metric='epoch')


# Select device
args.device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
print(f'Device is {args.device}')
device = torch.device(args.device)

# Create instance of the model
print(args.model)
modelC = getattr(bivae.models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)


# Load state_dict from training
print('Loading model {} from {}'.format(model.modelName, model_path))
model.load_state_dict(torch.load(model_path + '/model.pt'))
model._pz_params = model._pz_params


if not args.experiment:
    args.experiment = model.modelName

# set up run path
runId = datetime.datetime.now().isoformat()
runPath = Path(model_path + '/validate_'+runId)
runPath.mkdir(parents=True, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)


# Get the data
train_loader, test_loader, val_loader = model.getDataLoaders(args.batch_size, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")




# Define a sampler for generating new samples
# model.sampler = GaussianMixtureSampler()
model.sampler = None

# assesser = custom_mnist_fashion(model)
def eval():
    """Compute all metrics on the entire test dataset"""

    model.eval()

    b_metrics = {}

    for i, dataT in enumerate(tqdm(test_loader)):
        data = unpack_data(dataT, device=device)
        # update_dict_list(b_metrics, model.compute_conditional_likelihood(data, 1, 0, K= info.k))
        # update_dict_list(b_metrics, model.compute_conditional_likelihood(data, 0,1, K=info.k))
        update_dict_list(b_metrics, model.compute_conditional_likelihoods(data, K=info.k))
        update_dict_list(b_metrics, model.compute_joint_likelihood(data,K=info.k))

    # Get mean and standard deviation accross batches
    m_metrics, s_metrics = get_mean_std(b_metrics)
    wandb.log(m_metrics)
    wandb.log(s_metrics)
    print_mean_std(m_metrics,s_metrics)

    return



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        for r in range(1): # The number of independant runs
            eval()
