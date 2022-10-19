import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

import models
from models.samplers import GaussianMixtureSampler
from utils import Logger, Timer, unpack_data, update_dict_list, get_mean_std, print_mean_std

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--use-pretrain', type=str, default='')


# args
info = parser.parse_args()

# load args from disk if pretrained model path is given
pretrained_path = info.use_pretrain
with open(pretrained_path + 'args.json', 'r') as fcc_file:
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
experiment_name = args.experiment if args.experiment != '' else args.model
# wand_mode = 'online' if not args.eval_mode else 'disabled'
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
model.sampler = GaussianMixtureSampler()
# model.sampler = None

# Define the parameters for assessing quality
# assesser = Inception_quality_assess(model)
# assesser.check_activations(runPath)

# assesser = custom_mnist_fashion(model)
def eval():
    """Compute all metrics on the entire test dataset"""

    model.eval()
    # Compute all train latents
    model.compute_all_train_latents(train_loader)

    # re-fit the sampler before computing metrics
    if model.sampler is not None:
        model.sampler.fit_from_latents(model.train_latents[0])
    b_metrics = {}
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            # print(dataT)
            classes = dataT[0][1], dataT[1][1]

            update_dict_list(b_metrics, model.compute_metrics(data, runPath, epoch=2, classes=classes, freq=3, ns=30))
            if i == 0:
                model.sample_from_conditional(data, runPath, epoch=0)
                model.reconstruct(data, runPath, epoch=0)
                model.analyse(data, runPath, epoch=0, classes=classes)
                model.analyse_posterior(data, n_samples=10, runPath=runPath, epoch=0, ticks=None, N=100)
                model.generate(runPath, epoch=0, N=32, save=True)
                model.generate_from_conditional(runPath, epoch=0, N=32, save=True)

        for i in range(1):
            # Compute fids 10 times to have a std
            # update_dict_list(b_metrics,model.assess_quality(assesser,runPath))

            model.compute_fid(batch_size=50)

            # cond_gen_data = model.generate_from_conditional(runPath, 0)
            # np.save(f'{runPath}/cond_gen_data.npy',cond_gen_data.cpu().numpy() )


    m_metrics, s_metrics = get_mean_std(b_metrics)
    print_mean_std(m_metrics,s_metrics)

    return



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        eval()
