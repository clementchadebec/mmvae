# JMVAE_NF specification for MNIST-SVHN experiment

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import wandb
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from torchvision import transforms

from bivae.analysis.classifiers import load_pretrained_mnist, load_pretrained_svhn
from bivae.dataloaders import MNIST_SVHN_DL, BINARY_MNIST_SVHN_DL
from bivae.my_pythae.models import my_VAE, laplace_VAE
from bivae.utils import update_details
from bivae.vis import plot_hist
from .mmvae import MMVAE
from ..nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.analysis.accuracies import compute_accuracies
from bivae.utils import add_channels, unpack_data
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader
from ..modalities.mnist_svhn import fid

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

hidden_dim = 512




class MNIST_SVHN(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE


        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) # Standard MLP for
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(MNIST_SVHN, self).__init__(params, vaes)
        self.modelName = 'mmvae_mnist_svhn'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1) if params.llik_scaling == 0 else (params.llik_scaling, 1)
        self.shape_mods = [(1,28,28),(3,32,32)]
        
    def set_classifiers(self):
        
        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn()]
        


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val


    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""
        
        self.set_classifiers()
        general_metrics = MMVAE.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies

    def compute_fid(self, batch_size):
        return fid(self, batch_size)