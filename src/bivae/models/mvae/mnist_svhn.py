"MVAE specification for MNIST-SVHN"

# JMVAE_NF specification for MNIST-SVHN experiment


import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

from bivae.analysis import (compute_accuracies, load_pretrained_mnist,
                            load_pretrained_svhn)
from bivae.dataloaders import MNIST_SVHN_DL
from bivae.my_pythae.models import my_VAE
from bivae.utils import update_details

from ..modalities.mnist_svhn import fid
from ..nn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from .mvae import MVAE


class MNIST_SVHN(MVAE):
    def __init__(self, params):
        self.shape_mods = [(1,28,28),(3,32,32)]
        
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        
        vae = my_VAE
        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) 
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(MNIST_SVHN, self).__init__(params, vaes)
        self.modelName = 'mvae_mnist_svhn'
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1) if params.llik_scaling == 0 else (params.llik_scaling, 1)


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val



    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        
        self.set_classifiers()
        general_metrics = MVAE.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies




    def step(self, epoch):
        pass
    

    def set_classifiers(self):
        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn()]

    def compute_fid(self, batch_size):
        return fid(self, batch_size)


