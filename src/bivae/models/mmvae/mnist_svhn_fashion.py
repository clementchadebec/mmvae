# MMVAE specification for MNIST-SVHN-FASHION experiment

import torch
import torch.distributions as dist
import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

import wandb
from bivae.analysis.accuracies import compute_accuracies
from bivae.analysis.classifiers import (load_pretrained_fashion,
                                        load_pretrained_mnist,
                                        load_pretrained_svhn)
from bivae.dataloaders import MNIST_SVHN_FASHION_DL
from bivae.my_pythae.models import laplace_VAE, my_VAE
from bivae.utils import update_details

from ..modalities.trimodal import *
from ..nn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from .mmvae import MMVAE

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}


class MNIST_SVHN_FASHION(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae_config3 = vae_config((1,28,28), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE


        e1, e2,e3 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2), Encoder_VAE_MLP(vae_config3)
        d1, d2,d3 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2), Decoder_AE_MLP(vae_config3)

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=e1, decoder=d1),
            vae(model_config=vae_config2, encoder=e2, decoder=d2),
            vae(vae_config3,e3,d3)

        ])
        super(MNIST_SVHN_FASHION, self).__init__(params, vaes)
        self.modelName = 'mmvae_msf'

        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.vaes[2].modelName = 'fashion'
        self.shape_mods = [(1,28,28),(3,32,32),(1,28,28)]
        self.lik_scaling = ((3*32*32)/(1*28*28),1,(3*32*32)/(1*28*28)) if params.llik_scaling == 0 else (1,1,1)
        wandb.log({'lik_scaling' : self.lik_scaling})

    def set_classifiers(self):
        
        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn(), load_pretrained_fashion()]
        


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
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
        return fid(self,batch_size)