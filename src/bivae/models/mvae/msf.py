"MVAE specification for MNIST-SVHN-FASHION"


import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import wandb
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from torchvision import transforms

from bivae.analysis import MnistClassifier, SVHNClassifier
from bivae.dataloaders import MNIST_SVHN_FASHION_DL
from bivae.my_pythae.models import my_VAE
from bivae.utils import update_details
from bivae.vis import plot_hist
from .mvae import MVAE
from ..nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies, load_pretrained_fashion
from ..modalities.trimodal import *
from torchvision.utils import save_image





class MNIST_SVHN_FASHION(MVAE):
    def __init__(self, params):
        self.shape_mods = [(1,28,28),(3,32,32), (1,28,28)]
        
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae_config3 = vae_config((1,28,28), params.latent_dim)

        vae = my_VAE


        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) # Standard MLP for
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        encoder3, decoder3 = Encoder_VAE_MLP(vae_config3), Decoder_AE_MLP(vae_config3)
        
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2), 
            vae(model_config=vae_config3, encoder = encoder3, decoder = decoder3)
        ])
        
        super(MNIST_SVHN_FASHION, self).__init__(params, vaes)
        self.modelName = 'mvae_msf'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.vaes[2].modelName = 'fashion'
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1,(3 * 32 * 32) / (1 * 28 * 28)) if params.llik_scaling == 0 else (params.llik_scaling, 1)
        # self.lik_scaling = (1, 1,1) if params.llik_scaling == 0 else (params.llik_scaling, 1)
        self.subsampling = params.subsampling
        self.k_subsample = params.k_subsample
        self.subsets = np.array([np.array([1,2]), np.array([0,2]), np.array([0,1])])
        wandb.config.update(dict(subsampling = self.subsampling,
                                 k_subsample = self.k_subsample,
                                 ))

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val



    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""
        
        self.set_classifiers()
        general_metrics = MVAE.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        # Compute conditional accuracies
        cond_acc = compute_poe_subset_accuracy(self,data,classes,n_data,ns)
        update_details(accuracies, cond_acc)
        return accuracies




    def step(self, epoch):
        pass
    

    def set_classifiers(self):

        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn(), load_pretrained_fashion()]

    def compute_fid(self, batch_size):
        return fid(self, batch_size)


    def sample_from_poe(self, data, runPath, epoch, n=10):
        sample_from_poe_vis(self, data, runPath, epoch, n)

            

    def compute_conditional_likelihoods(self, data, K=1000, batch_size_K=100):
        d =  super().compute_conditional_likelihoods(data, K, batch_size_K)
        
        poe_ll = compute_all_cond_ll_from_poe_subsets(self,data,K,batch_size_K)
        update_details(d,poe_ll)
        
        return d