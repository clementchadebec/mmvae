# Toy example with cercles and discs

import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

from vis import plot_embeddings, plot_kls_df, plot_posteriors
from .mmvae import MMVAE
from .vae_circles import CIRCLES


data_path = '../data/circles_and_discs/'

class CIRCLES_DISCS(MMVAE):
    def __init__(self, params):
        super(CIRCLES_DISCS, self).__init__(params, CIRCLES, CIRCLES)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'circles_discs'

        self.vaes[0].modelName = 'circles'
        self.vaes[1].modelName = 'discs'
        self.data_path = params.data_path

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        if not (os.path.exists(self.data_path + 'circles_train.pt')
                and os.path.exists(self.data_path + 'discs_train.pt')
                and os.path.exists(self.data_path + 'circles_test.pt')
                and os.path.exists(self.data_path + 'discs_test.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, 'circles', shuffle, device, data_path=self.data_path)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size,'discs', shuffle, device, data_path=self.data_path)

        train_circles_discs = TensorDataset([t1.dataset,t2.dataset])
        test_circles_discs = TensorDataset([s1.dataset, s2.dataset])


        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train, test

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(CIRCLES_DISCS, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(CIRCLES_DISCS, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def sample_from_conditional(self,data, runPath,epoch,n = 10):

        recons = super(CIRCLES_DISCS, self).sample_from_conditional([d[:8] for d in data], n=n)

        for r, recon_list in enumerate(recons):
            for o, recon in enumerate(recon_list):
                _data = data[r][:8].cpu()
                recon = torch.reshape(recon,(n*8,1,32,32)).cpu()
                comp = torch.cat([_data,recon])
                save_image(comp, '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch, ticks=None):
        zemb, zsl, kls_df = super(CIRCLES_DISCS, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch), ticks=ticks, K=10)
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

    def analyse_posterior(self,data, n_samples,runPath, epoch, ticks=None):
        means, stds = super(CIRCLES_DISCS, self).analyse_posterior(data, n_samples)
        plot_posteriors(means, stds, '{}/posteriors_{:03}.png'.format(runPath,epoch), ticks=ticks)

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
