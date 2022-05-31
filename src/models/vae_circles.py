# circles and discs model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from datasets import CIRCLES_DATASET
from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE


# Constants
data_path = '../data/circles_and_discs/'
dataSize = torch.Size([1, 32, 32])
data_dim = int(prod(dataSize))
hidden_dim = 512
dist_dict = {'normal' : dist.Normal, 'laplace' : dist.Laplace}


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for image data. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        lv = torch.exp(0.5*self.fc22(e))
        return self.fc21(e), lv + Constants.eta

# class Enc(nn.Module):
#     """ Generate latent parameters for image data. """
#
#     def __init__(self, latent_dim, num_hidden_layers=1):
#         super(Enc, self).__init__()
#         module1 = []
#         module1.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
#         module1.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
#         self.enc1 = nn.Sequential(*module1)
#         self.fc1_1 = nn.Linear(hidden_dim, latent_dim//2)
#         self.fc1_2 = nn.Linear(hidden_dim, latent_dim//2)
#
#         module2 = []
#         module2.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
#         module2.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
#         self.enc2 = nn.Sequential(*module2)
#         self.fc2_1 = nn.Linear(hidden_dim, latent_dim//2)
#         self.fc2_2 = nn.Linear(hidden_dim, latent_dim//2)
#
#
#     def forward(self, x):
#         y1 = self.enc1(x.view(*x.size()[:-3], -1))
#         y2 = self.enc2(x.view(*x.size()[:-3], -1))
#         mu1, lv1 = self.fc1_1(y1), self.fc1_2(y1)
#         mu2, lv2 = self.fc2_1(y2), self.fc2_2(y2)
#         return torch.cat([mu1, mu2], dim=1), torch.exp(0.5*torch.cat([lv1, lv2], dim=1))
#
#
class Dec(nn.Module):
    """ Generate an circle/disc image given a sample from the latent space. """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *dataSize))  # reshape data
        d = d.clamp(Constants.eta, 1 - Constants.eta)

        return d, torch.tensor(0.75).to(z.device)  # mean, length scale


class CIRCLES(VAE):
    """ Derive a specific sub-class of a VAE for circles and dics dataset. """

    def __init__(self, params):
        super(CIRCLES, self).__init__(
            dist_dict[params.dist],  # prior
            dist_dict[params.dist],  # likelihood
            dist_dict[params.dist],  # posterior
            Enc(params.latent_dim, params.num_hidden_layers),
            Dec(params.latent_dim, params.num_hidden_layers),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'circles' # circles or discs
        self.dataSize = dataSize
        self.llik_scaling = 1.
        self.data_path = params.data_path

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size, type = 'circles', shuffle=True, device="cuda",data_path=data_path ):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        # create datasets
        train = DataLoader(CIRCLES_DATASET(data_path + type + '_train.pt', data_path + 'labels_train.pt', data_path + 'r_' + type+'_train.pt'),batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(CIRCLES_DATASET(data_path + type + '_test.pt', data_path + 'labels_test.pt', data_path + 'r_'+type+'_test.pt'),batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(CIRCLES, self).generate(N, K).cpu()
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(CIRCLES, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon]).data.cpu()
        save_image(comp, '{}/recon_{:03d}.png'.format(runPath, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(CIRCLES, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
