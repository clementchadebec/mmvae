# JMVAE_NF specification for CelebA experiment

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn
from torchvision.utils import save_image
import pythae
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from bivae.models.nn import Encoder_VAE_SVHN
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from bivae.utils import extract_rayon
from bivae.dataloaders import CELEBA_DL
import torch.nn.functional as F

from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies
from bivae.dcca.models import load_dcca_mnist_svhn

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

# Define the classifiers for analysis


class JMVAE_NF_CELEBA(JMVAE_NF):

    shape_mod1, shape_mod2 = (3, 64, 64), (1,1,40)
    modelName = 'jmvae_nf_dcca_celeb_a'


    def __init__(self, params):

        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig

        # Define the joint encoder
        hidden_dim = 1024
        pre_configs = [VAEConfig(self.shape_mod1, 128), VAEConfig(self.shape_mod2, 40)]
        joint_encoder = DoubleHeadJoint(hidden_dim,pre_configs[0], pre_configs[1],
                                        Encoder_ResNet_VAE_CELEBA ,
                                        Encoder_VAE_MLP,
                                        params)

        # Define the unimodal encoders config
        vae_config1 = vae_config(self.shape_mod1, params.latent_dim)
        vae_config2 = vae_config(self.shape_mod2, params.latent_dim)

        # # First load the DCCA encoders
        # self.dcca = load_dcca_mnist_svhn()
        # # Then add the flows
        # encoder1 = TwoStepsEncoder(self.dcca[0], params)
        # encoder2 = TwoStepsEncoder(self.dcca[1], params)
        encoder1 = Encoder_ResNet_VAE_CELEBA(vae_config1)
        encoder2 = Encoder_VAE_MLP(vae_config2)

        # Define the decoders
        decoder1, decoder2 = Decoder_ResNet_AE_CELEBA(vae_config1), Decoder_AE_MLP(vae_config2)
        # decoder1 = TwoStepsDecoder(Decoder_AE_MNIST,pre_configs[0], pretrained_decoders[0], params)
        # decoder2 = TwoStepsDecoder(Decoder_VAE_SVHN, pre_configs[1], pretrained_decoders[1], params)
        # decoder1, decoder2 = None, None

        # Then define the vaes
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)
        ])

        super(JMVAE_NF_CELEBA, self).__init__(params, joint_encoder, vaes)

        self.vaes[0].modelName = 'celeb'
        self.vaes[1].modelName = 'attributes'
        self.lik_scaling = ( np.prod(self.shape_mod2) / np.prod(self.shape_mod1), 1) if params.llik_scaling == 0.0 else (
        params.llik_scaling, 1)

        # Set the classifiers
        # self.classifier1, self.classifier2 = classifier1, classifier2

        self.recon_losses = ['mse', 'bce']

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = None):
        train, test, val = CELEBA_DL(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test, val



















