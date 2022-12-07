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

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--model', type=str, default='')


# args
info = parser.parse_args()

# load args from disk if pretrained model path is given
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

# Log parameters of the experiments
experiment_name = args.wandb_experiment if hasattr(args,'wandb_experiment') else args.model
# wand_mode = 'online' if not args.eval_mode else 'disabled'
wandb.init(project = experiment_name , entity="clementchadebec") # mode = ['online', 'offline', 'disabled']
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


train_loader, test_loader, val_loader = model.getDataLoaders(2000, device=device)
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
    # Compute all train latents
    # model.compute_all_train_latents(train_loader)

    # re-fit the sampler before computing metrics
    if model.sampler is not None:

        model.sampler.fit_from_latents(model.train_latents[0])
    b_metrics = {}

    for i, dataT in enumerate(tqdm(test_loader)):
        data = unpack_data(dataT, device=device)
        # print(dataT)
        classes = dataT[0][1], dataT[1][1]
        # Compute the classification accuracies
        update_dict_list(b_metrics, model.compute_metrics(data, runPath, epoch=2, classes=classes,n_data='all', freq=3, ns=1))
        if i == 0:
            model.sample_from_conditional(data, runPath, epoch=0)
            # try: 
            #     model.sample_from_poe(data, runPath, 0, n=10, divide_prior=True)
            # except:
            #     print('No function implemented for poe generation')
            
            # model.reconstruct(data, runPath, epoch=0)
            # model.analyse(data, runPath, epoch=0, classes=classes)
            # model.analyse_posterior(data, n_samples=10, runPath=runPath, epoch=0, ticks=None, N=100)
            model.generate(runPath, epoch=0, N=32, save=True)
            model.generate_from_conditional(runPath, epoch=0, N=32, save=True)


    update_dict_list(b_metrics, model.compute_fid(batch_size=256))




    m_metrics, s_metrics = get_mean_std(b_metrics)
    # wandb.log(m_metrics)
    # wandb.log(s_metrics)
    print_mean_std(m_metrics,s_metrics)

    return m_metrics



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        run_metrics = {}
        for r in range(5): # 5 independants runs to have a std deviation on each value
            print(f'Run {r} of the evaluation')
            update_dict_list(run_metrics, eval()) 
        m_run_metrics, s_run_metrics = get_mean_std(run_metrics)
        print_mean_std(m_run_metrics,s_run_metrics)
        wandb.log(m_run_metrics)
        wandb.log(s_run_metrics)
        
        