# JMVAE_NF specification for CelebA experiment

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms
from bivae.dataloaders import BasicDataset
from bivae.analysis.pytorch_fid.fid_score import calculate_frechet_distance

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn
from torchvision.utils import save_image
import pythae
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF, my_VAE_MAF, VAE_MAF_Config
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from bivae.models.nn import Encoder_VAE_SVHN
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from bivae.dataloaders import CELEBA_DL
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from bivae.utils import unpack_data, kl_divergence, add_channels, adjust_shape, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples

from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies
from bivae.dcca.models import load_dcca_celeba
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Decoder_VAE_SVHN, TwoStepsDecoder, TwoStepsEncoder
from torchvision.transforms import ToTensor
from bivae.analysis.classifiers.CelebA_classifier import load_celeba_classifiers
from bivae.analysis.pytorch_fid.inception import wrapper_inception
from bivae.analysis.pytorch_fid.fid_score import get_activations
from bivae.models.modalities.celeba import *


# Define the classifiers for analysis


class JMVAE_NF_CELEBA(JMVAE_NF):

    shape_mods = [(3, 64, 64), (1,1,40)]
    modelName = 'jmvae_nf_dcca_celeb_a'


    def __init__(self, params):
        
        if params.no_nf :
            vae_config, vae = VAEConfig , my_VAE
        else :
            vae_config = VAE_IAF_Config if params.flow == 'iaf' else VAE_MAF_Config
            vae = my_VAE_IAF if params.flow == 'iaf' else my_VAE_MAF

        # Define the joint encoder
        hidden_dim = 1024
        pre_configs = [VAEConfig(self.shape_mods[0], 128), VAEConfig(self.shape_mods[1], 40)]
        joint_encoder = DoubleHeadJoint(hidden_dim,pre_configs[0], pre_configs[1],
                                        Encoder_ResNet_VAE_CELEBA ,
                                        Encoder_VAE_MLP,
                                        params)

        # Define the unimodal encoders config
        vae_config1 = vae_config(self.shape_mods[0], params.latent_dim)
        vae_config2 = vae_config(self.shape_mods[1], params.latent_dim)

        # # First load the DCCA encoders
        if params.dcca :
            print("Preparing DCCA encoders")
            self.dcca = load_dcca_celeba()
            # # Then add the flows
            encoder1 = TwoStepsEncoder(self.dcca[0], params, hidden_dim=40, num_hidden=3)
            encoder2 = TwoStepsEncoder(self.dcca[1], params, hidden_dim=40,num_hidden=3)
        else :
            print("Preparing non DCCA encoders")
            encoder1 = Encoder_ResNet_VAE_CELEBA(vae_config1)
            encoder2 = Encoder_VAE_MLP(vae_config2)

        # Define the decoders
        decoder1, decoder2 = Decoder_ResNet_AE_CELEBA(vae_config1), Decoder_AE_MLP(vae_config2)
        # decoder1 = TwoStepsDecoder(Decoder_AE_MNIST,pre_configs[0], pretrained_decoders[0], params)
        # decoder2 = TwoStepsDecoder(Decoder_VAE_SVHN, pre_configs[1], pretrained_decoders[1], params)
        # decoder1, decoder2 = None, None

        # Then define the vaes
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)
        ])

        super(JMVAE_NF_CELEBA, self).__init__(params, joint_encoder, vaes)

        self.vaes[0].modelName = 'celeb'
        self.vaes[1].modelName = 'attributes'
        self.lik_scaling = ( np.prod(self.shape_mods[1]) / np.prod(self.shape_mods[0]), 1) if params.llik_scaling == 0.0 else (
        params.llik_scaling, 1)

        # Set the classifiers
        # self.classifier1, self.classifier2 = classifier1, classifier2

    
    def set_classifiers(self):
        self.classifiers = load_celeba_classifiers()


    def sample_from_conditional(self, data, runPath, epoch, n=10):
        return sample_from_conditional_celeba(self,data, runPath, epoch, n)


    
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

    
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = ToTensor()):
        train, test, val = CELEBA_DL(self.data_path).getDataLoaders(batch_size, shuffle, device,transform=transform)
        return train, test, val


    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """

        inputs :

        - classes of shape (batch_size, 40)"""
        self.set_classifiers()
        metrics = compute_accuracies(self, data, runPath, epoch, classes, n_data, ns, freq)
        general_metrics = {}
        #general_metrics = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)

        update_details(metrics, general_metrics)
        return metrics


    def compute_fid(self, batch_size):
        return compute_fid_celeba(self, batch_size)