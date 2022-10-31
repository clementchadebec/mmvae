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

import numpy as np
import torch
from torch import optim

import wandb
import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, update_details, extract_rayon\
    ,load_joint_vae, update_dict_list, get_mean_std, print_mean_std
from vis import plot_hist
from models.samplers import GaussianMixtureSampler
from bivae.analysis import compute_accuracies

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--config-path', type=str, default='')


# args
info = parser.parse_args()

# load args from disk if pretrained model path is given
with open(info.config_path, 'r') as fcc_file:
    args = argparse.Namespace()
    args.__dict__.update(json.load(fcc_file))

learning_rate = 1e-3
# Log parameters of the experiments
experiment_name = args.wandb_experiment
wandb.init(project = experiment_name , entity="asenellart", config={'lr' : learning_rate}, mode=args.wandb_mode) 
wandb.config.update(args)
wandb.define_metric('epoch')
wandb.define_metric('*', step_metric='epoch')

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# load args from disk if pretrained model path is given
use_pretrain = args.use_pretrain != ''
pretrained_path = args.use_pretrain
if use_pretrain :
    old_args = torch.load(pretrained_path + 'args.rar')
    args, new_args = old_args, args
    min_epoch = args.epochs
    args.epochs = min_epoch + new_args.epochs
    args.warmup = min_epoch + new_args.warmup
    args.freq_analytics = new_args.freq_analytics

args.device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
print(f'Device is {args.device}')
device = torch.device(args.device)

# load model
print(args.model)
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

skip_warmup = args.skip_warmup
# pretrained_joint_path = '../experiments/jmvae_nf_mnist/2022-06-15/2022-06-15T09:53:04.9472623yb2z1h0/'
# pretrained_joint_path = '../experiments/jmvae_nf_circles_squares/2022-06-14/2022-06-14T16:02:13.698346trcaealp/'
# pretrained_joint_path = '../experiments/clean_mnist_svhn/2022-06-29/2022-06-29T11:41:41.132687__5qri92/'
# pretrained_joint_path = '../experiments/jmvae/2022-06-28/2022-06-28T17:25:01.03903846svjh2d/'
pretrained_joint_path = '../experiments/celeba/2022-10-13/2022-10-13T13:54:42.595068mmpybk9u/'

min_epoch = 1

if skip_warmup:
    print('Loading joint encoder and decoders')
    load_joint_vae(model,pretrained_joint_path)
    min_epoch = args.warmup

if use_pretrain:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.pt'))
    model._pz_params = model._pz_params


if not args.experiment:
    args.experiment = model.modelName

# set up run path
runId = datetime.datetime.now().isoformat()

experiment_dir = Path('../experiments/' + args.experiment + '/' + datetime.date.today().isoformat())
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=learning_rate, amsgrad=True)
train_loader, test_loader, val_loader = model.getDataLoaders(args.batch_size, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")


# Objective function to use on train data
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))

# Objective function to use on test data
t_objective = objective

# Define a sampler for generating new samples
model.sampler = GaussianMixtureSampler()

def train(epoch, agg):
    model.train()
    b_loss = 0
    b_details = {}
    for i, dataT in enumerate(tqdm(train_loader)):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss, details = objective(model, data,args.K,epoch,args.warmup, args.beta_prior)
        loss = -loss # minimization
        loss.backward()
        optimizer.step()

        b_loss += loss.item()
        update_details(b_details, details)
        # print('after update_det',b_details['loss'])
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f} details : {}".format(i, loss.item() / args.batch_size, b_details['loss_0']/b_details['loss_1']))
    b_details = {k + '_train': b_details[k]/len(train_loader.dataset) for k in b_details.keys()}
    wandb.log(b_details)
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}, details : {}'.format(epoch, agg['train_loss'][-1], b_details))

    # Change the value of the parameters
    model.step(epoch)

def test(epoch, agg):
    model.eval()

    # re-fit the sampler before computing metrics
    if model.sampler is not None:
        # Compute all train latents
        model.compute_all_train_latents(train_loader)
        model.sampler.fit_from_latents(model.train_latents[0])
    b_loss = 0
    b_details = {}
    with torch.no_grad():
        for i, dataT in enumerate(val_loader):
            data = unpack_data(dataT, device=device)
            classes = dataT[0][1], dataT[1][1]
            ticks = np.arange(len(data[0])) #or simply the indexes
            loss, details = t_objective(model, data, K=args.K, beta_prior = args.beta_prior, epoch=epoch,warmup=args.warmup)
            loss = -loss
            b_loss += loss.item()
            update_details(b_details, details)
            if i == 0:
                wandb.log({'epoch' : epoch})
                # Compute accuracies
                # wandb.log(model.compute_metrics(data, runPath, epoch, classes))
                model.sample_from_conditional(data, runPath,epoch)
                model.reconstruct(data, runPath, epoch)
                if not args.no_analytics and (epoch%args.freq_analytics == 0 or epoch==1):
                    # model.analyse(data, runPath, epoch, classes=classes)
                    # model.analyse_posterior(data, n_samples=10, runPath=runPath, epoch=epoch, ticks=ticks, N=100)
                    model.generate(runPath, epoch, N=32, save=True)
                    # model.generate_from_conditional(runPath, epoch, N=32, save=True)
                    if args.model in ['circles_discs','j_circles_discs', 'jnf_circles_squares', 'circles_squares'] :
                        if epoch == 1:
                            print("Computing test histogram")
                            plot_hist(extract_rayon(data[0].unsqueeze(1)), runPath + '/hist_test_0.png')
                            plot_hist(extract_rayon(data[1].unsqueeze(1)), runPath + '/hist_test_1.png')
                        model.analyse_rayons(data, dataT[0][2],dataT[1][2],runPath, epoch, [dataT[0][1], 1-dataT[0][1]])

    b_details = {k + '_test': b_details[k] / len(val_loader.dataset) for k in b_details.keys()}
    wandb.log(b_details)
    agg['test_loss'].append(b_loss / len(val_loader.dataset))
    wandb.log({'test_loss' : b_loss / len(val_loader.dataset) })
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))
    return b_loss / len(val_loader.dataset)




def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in val_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -t_objective(model, data, K).item()

    marginal_loglik /= len(val_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        best_loss = torch.inf
        for epoch in range(min_epoch, args.epochs + 1):
            if epoch == args.warmup :
                print(f" ====> Epoch {epoch} Reset the optimizer")
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)

            train(epoch, agg)
            test_loss = test(epoch, agg)
            if test_loss < best_loss:
                save_model(model, runPath + '/model.pt')
                print("Saved model after improvement of {}".format(best_loss-test_loss))
                best_loss = test_loss

            save_vars(agg, runPath + '/losses.pt')

        if args.logp:  # compute as tight a marginal likelihood as possible
            estimate_log_marginal(5000)
