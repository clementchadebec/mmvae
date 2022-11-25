# MVAE specification for CelebA experiment

import torch
import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.nn.benchmarks.celeba import (Decoder_ResNet_AE_CELEBA,
                                                Encoder_ResNet_VAE_CELEBA)
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

import wandb
from bivae.analysis.classifiers.CelebA_classifier import \
    load_celeba_classifiers
from bivae.dataloaders import CELEBA_DL
from bivae.models.modalities.celeba import *
from bivae.my_pythae.models import my_VAE
from bivae.utils import adjust_shape, update_details
from bivae.vis import save_samples

from .mvae import MVAE


class celeba(MVAE):
    def __init__(self, params):


        self.shape_mods=[(3,64,64),(1,1,40)]

        vae_config = VAEConfig
        vae_config1 = vae_config((3,64,64), params.latent_dim)
        vae_config2 = vae_config((1,1,40), params.latent_dim)
        

        encoder1, encoder2 = Encoder_ResNet_VAE_CELEBA(vae_config1), Encoder_VAE_MLP(vae_config2) 
        decoder1, decoder2 = Decoder_ResNet_AE_CELEBA(vae_config1), Decoder_AE_MLP(vae_config2)
        vae = my_VAE
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(celeba, self).__init__(params, vaes)
        self.modelName = 'mvae_celeba'
        self.vaes[0].modelName = 'celeb'
        self.vaes[1].modelName = 'attributes'
        self.lik_scaling = (1,50) if params.llik_scaling == 0 else (1, params.llik_scaling) # settings mentioned in the paper
        wandb.config.update({'lik_scalings' : self.lik_scaling})

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = CELEBA_DL(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test, val





    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=300, freq=10):
        self.set_classifiers()
        metrics = compute_accuracies(self, data, runPath, epoch, classes, n_data, ns, freq)

        general_metrics = MVAE.compute_metrics(self, runPath, epoch, freq=freq)

        update_details(metrics, general_metrics)
        return metrics


    
    def generate(self,runPath, epoch, N= 8, save=False):
        """Generate samples from sampling the prior distribution"""
        self.eval()
        with torch.no_grad():
            data = []
            if self.sampler is None:
                pz = self.pz(*self.pz_params)
                latents = pz.rsample(torch.Size([N])).squeeze()
            else :
                latents = self.sampler.sample(num_samples=N)
            for d, vae in enumerate(self.vaes):
                data.append(vae.decoder(latents)["reconstruction"])

        if save:
            data = [*adjust_shape(data[0],attribute_array_to_image(data[1]))]
            file = ('{}/generate_{:03d}'+self.save_format).format(runPath, epoch)
            save_samples(data,file)
            wandb.log({'generate_joint' : wandb.Image(file)})
        return data  # list of generations---one for each modality

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        return sample_from_conditional_celeba(self, data, runPath, epoch, n)



    def set_classifiers(self):

        # Define the classifiers for analysis
        self.classifiers = load_celeba_classifiers()


    def step(self, epoch):
        pass



    def compute_fid(self,batch_size):
        return compute_fid_celeba(self, batch_size)










