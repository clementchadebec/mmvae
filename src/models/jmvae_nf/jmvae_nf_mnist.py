# JMVAE_NF specification for MNIST

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist

from utils import get_mean, kl_divergence
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from utils import extract_rayon
from ..dataloaders import MNIST_FASHION_DATALOADER

from ..vae_circles import CIRCLES
from ..joint_encoders import DoubleHeadMLP
from ..jmvae_nf import JMVAE_NF

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,28,28)

vae = my_VAE_IAF
vae_config = VAE_IAF_Config

Enc = DoubleHeadMLP
hidden_dim = 512


class JMVAE_NF_MNIST(JMVAE_NF):
    def __init__(self, params):
        joint_encoder = Enc(28*28,28*28,hidden_dim,params.latent_dim, params.num_hidden_layers)
        vae = my_VAE_IAF
        vae_config = VAE_IAF_Config(input_dim, params.latent_dim)
        super(JMVAE_NF_MNIST, self).__init__(params, joint_encoder, vae, vae_config)
        self.modelName = 'jmvae_nf_mnist'
        self.data_path = params.data_path


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        train, test = MNIST_FASHION_DATALOADER(self.data_path).getDataLoaders(batch_size,shuffle,device)
        return train, test


    def sample_from_conditional(self, data, runPath, epoch, n=10):
        JMVAE_NF.sample_from_conditional(self,data, runPath, epoch,n)





