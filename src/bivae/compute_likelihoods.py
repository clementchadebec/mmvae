import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import random
from tqdm import tqdm
from copy import deepcopy

from analysis.pytorch_fid import wrapper_inception
from analysis import Inception_quality_assess, custom_mnist_fashion

import numpy as np
import torch
from torch import optim
from torchvision import transforms


import wandb
import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, update_details, extract_rayon, add_channels,load_joint_vae, update_dict_list, get_mean_std, print_mean_std
from vis import plot_hist
from models.samplers import GaussianMixtureSampler

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--use-pretrain', type=str, default='')


# args
info = parser.parse_args()

# load args from disk if pretrained model path is given
pretrained_path = info.use_pretrain
args = torch.load(pretrained_path + 'args.rar')

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Log parameters of the experiments
experiment_name = args.experiment if args.experiment != '' else args.model
wand_mode = 'disabled'
wandb.init(project = experiment_name , entity="asenellart", mode='disabled') # mode = ['online', 'offline', 'disabled']
wandb.config.update(args)
wandb.define_metric('epoch')
wandb.define_metric('*', step_metric='epoch')



args.device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
print(f'Device is {args.device}')
device = torch.device(args.device)

# Create instance of the model
print(args.model)
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)


# Load state_dict from training
print('Loading model {} from {}'.format(model.modelName, pretrained_path))
model.load_state_dict(torch.load(pretrained_path + '/model.pt'))
model._pz_params = model._pz_params


if not args.experiment:
    args.experiment = model.modelName

# set up run path
runId = datetime.datetime.now().isoformat()

runPath = Path(pretrained_path + '/validate_'+runId)
runPath.mkdir(parents=True, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)



train_loader, test_loader, val_loader = model.getDataLoaders(args.batch_size, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")




# Define a sampler for generating new samples
# model.sampler = GaussianMixtureSampler()
model.sampler = None

# Define the parameters for assessing quality
# assesser = Inception_quality_assess(model)
# assesser.check_activations(runPath)

# assesser = custom_mnist_fashion(model)
def eval():
    """Compute all metrics on the entire test dataset"""

    model.eval()

    b_metrics = {}
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            update_dict_list(b_metrics, model.compute_conditional_likelihoods(data, K=200))


    m_metrics, s_metrics = get_mean_std(b_metrics)
    print_mean_std(m_metrics,s_metrics)

    return



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        eval()
