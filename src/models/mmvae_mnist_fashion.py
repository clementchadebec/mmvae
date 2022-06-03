# MNIST-SVHN multi-modal model specification
import os
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid
from .dataloaders import MNIST_FASHION_DATALOADER

from vis import plot_embeddings, plot_kls_df, plot_posteriors
from .mmvae import MMVAE
from .vae_mnist import MNIST
from .vae_svhn import SVHN
from utils import tensor_classes_labels

class MNIST_FASHION(MMVAE):
    def __init__(self, params):
        super(MNIST_FASHION, self).__init__(params, MNIST, MNIST)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'mnist-fashion'
        self.data_path = params.data_path
        print(self.data_path)

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        train,test = MNIST_FASHION_DATALOADER(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(MNIST_FASHION, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(MNIST_FASHION, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def sample_from_conditional(self,data, runPath,epoch,n = 10):

        recons = super(MNIST_FASHION, self).sample_from_conditional([d[:8] for d in data], n=n)

        for r, recon_list in enumerate(recons):
            for o, recon in enumerate(recon_list):
                _data = data[r][:8].cpu()
                recon = torch.reshape(recon,(n*8,1,28,28)).cpu()
                comp = torch.cat([_data,recon])
                save_image(comp, '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))


    def analyse(self, data, runPath, epoch, classes = None, ticks=None):
        zemb, zsl, kls_df = super(MNIST_FASHION, self).analyse(data, K=1)
        labels = [ *[vae.modelName.lower() for vae in self.vaes]]
        zsl, labels = tensor_classes_labels(zsl, 2 * list(classes), labels, classes.unique().numpy().astype(str)) if classes \
                                                                                                  is not None else (zsl, labels)

        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch), ticks = ticks, K=1)
        # plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))

    def analyse_posterior(self,data, n_samples,runPath, epoch, ticks=None):
        means, stds = super(MNIST_FASHION, self).analyse_posterior(data, n_samples)
        plot_posteriors(means, stds, '{}/posteriors_{:03}.png'.format(runPath,epoch),labels=['numbers', 'fashion'], ticks=ticks)
        return means, stds

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
