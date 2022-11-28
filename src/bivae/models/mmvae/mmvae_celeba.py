# MMVAE specification for CelebA experiment

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from pythae.models import VAE_IAF_Config, VAE_LinNF_Config, VAEConfig
from pythae.models.nn.benchmarks.celeba import (Decoder_ResNet_AE_CELEBA,
                                                Encoder_ResNet_VAE_CELEBA)
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import wandb
from bivae.analysis.classifiers.CelebA_classifier import \
    load_celeba_classifiers
from bivae.analysis.pytorch_fid import (calculate_frechet_distance,
                                        wrapper_inception)
from bivae.dataloaders import CELEBA_DL, BasicDataset
from bivae.models.modalities.celeba import *
from bivae.my_pythae.models import (laplace_VAE, my_VAE, my_VAE_IAF,
                                    my_VAE_LinNF)
from bivae.utils import (add_channels, adjust_shape, get_mean, kl_divergence,
                         negative_entropy, unpack_data, update_details)
from bivae.vis import (plot_embeddings_colorbars, plot_hist,
                       plot_samples_posteriors, save_samples,
                       save_samples_mnist_svhn, tensors_to_df)

from ..nn import (Decoder_AE_MNIST, Decoder_VAE_SVHN, DoubleHeadJoint,
                  DoubleHeadMLP, Encoder_VAE_MNIST, Encoder_VAE_SVHN)
from .mmvae import MMVAE


class celeba(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig

        self.shape_mods = [(3,64,64),(1,1,40)]

        vae_config1 = vae_config((3,64,64), params.latent_dim)
        vae_config2 = vae_config((1,1,40), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE

        encoder1, encoder2 = Encoder_ResNet_VAE_CELEBA(vae_config1), Encoder_VAE_MLP(vae_config2) # Standard MLP for
        decoder1, decoder2 = Decoder_ResNet_AE_CELEBA(vae_config1), Decoder_AE_MLP(vae_config2)

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(celeba, self).__init__(params, vaes)
        self.modelName = 'mmvae_celeba'

        self.vaes[0].modelName = 'celeb'
        self.vaes[1].modelName = 'attributes'
        self.lik_scaling = (np.prod(self.shape_mods[1]) / np.prod(self.shape_mods[0]),1) if params.llik_scaling == 0 else (params.llik_scaling, 1)

        wandb.config.update({'lik_scalings' : self.lik_scaling})

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = CELEBA_DL(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test, val





    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=300, freq=10):
        self.set_classifiers()
        metrics = compute_accuracies(self, data, runPath, epoch, classes, n_data, ns, freq)
        general_metrics = MMVAE.compute_metrics(self, runPath, epoch, freq=freq)

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
        return sample_from_conditional_celeba(self, data, runPath, epoch,n)

    def analyse(self, data, runPath, epoch=0, classes=None):
        return


    def compute_fid(self, batch_size):
        return compute_fid_celeba(self, batch_size)


    def set_classifiers(self):

        # Define the classifiers for analysis
        self.classifiers = load_celeba_classifiers()    










