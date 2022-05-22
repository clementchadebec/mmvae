import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from torch import optim

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, update_details

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='mnist_svhn', metavar='M',
                    choices=[s[4:] for s in dir(models) if 'VAE_' in s],
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=['elbo', 'iwae', 'dreg', 'vaevae_w2', 'vaevae_kl', 'jmvae', 'multi_elbos', 'svae', 'telbo'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 20)')
parser.add_argument('--looser', action='store_true', default=False,
                    help='use the looser version of IWAE/DREG')
parser.add_argument('--llik_scaling', type=float, default=0.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
                    help='estimate tight marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-analytics', action='store_true', default=False,
                    help='disable plotting analytics')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dist', type=str, default = 'normal',
                    choices= ['normal', 'laplace'])
parser.add_argument('--beta', type=float, default=1000,
                    help='scaling factor for the regularization in vaevae loss')
parser.add_argument('--data-path', type=str, default = '../data/')

parser.add_argument('--warmup', type=int, default=0)

parser.add_argument('--beta-prior', type=float, default = 1)

# args
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
pretrained_path = ""
# pretrained_path = "../../mnist-svhn"
if args.pre_trained:
    pretrained_path = args.pre_trained
    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'Cuda is {args.cuda}')
device = torch.device("cuda" if args.cuda else "cpu")

# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)


if pretrained_path:
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
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
                       lr=1e-3, amsgrad=True)
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)

# Objective function to use on train data
objective = getattr(objectives,
                    ('m_' if hasattr(model, 'vaes') else '')
                    + args.obj
                    + ('_looser' if (args.looser and args.obj != 'elbo') else ''))

# Objective fulrnction to use on test data
# t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'elbo')
t_objective = objective

def train(epoch, agg):
    model.train()
    b_loss = 0
    b_details = {}
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
        optimizer.zero_grad()
        loss, details = objective(model, data,args.K, args.beta,epoch,args.warmup, args.beta_prior)
        loss = -loss # minimization
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        update_details(b_details, details)
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f} details : {}".format(i, loss.item() / args.batch_size, b_details['loss_0']/b_details['loss_1']))
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}, details : {}'.format(epoch, agg['train_loss'][-1], b_details))


def test(epoch, agg):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data = unpack_data(dataT, device=device)
            classes = dataT[0][1]
            ticks = np.arange(len(data[0])) #or simply the indexes
            loss, details = t_objective(model, data, K=args.K, beta = args.beta, beta_prior = args.beta_prior)
            loss = -loss
            b_loss += loss.item()
            if i == 0:
                model.sample_from_conditional(data, runPath,epoch)
                model.reconstruct(data, runPath, epoch)
                if not args.no_analytics:
                    model.analyse(data, runPath, epoch, classes=classes)
                    model.analyse_posterior(data, n_samples=8, runPath=runPath, epoch=epoch, ticks=ticks)

                    if args.model in ['circles_discs','j_circles_discs'] :
                        model.analyse_rayons(data, dataT[0][2],dataT[1][2],runPath, epoch)
    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


def estimate_log_marginal(K):
    """Compute an IWAE estimate of the log-marginal likelihood of test data."""
    model.eval()
    marginal_loglik = 0
    with torch.no_grad():
        for dataT in test_loader:
            data = unpack_data(dataT, device=device)
            marginal_loglik += -t_objective(model, data, K).item()

    marginal_loglik /= len(test_loader.dataset)
    print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)
            test(epoch, agg)
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            model.generate(runPath, epoch)
        if args.logp:  # compute as tight a marginal likelihood as possible
            estimate_log_marginal(5000)
