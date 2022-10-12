# JMVAE_NF specification for MNIST-CONTOUR

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF

from bivae.dataloaders import MNIST_CONTOUR_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, TwoStepsEncoder
from bivae.dcca.models import load_dcca_mnist_contour

from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import MnistClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1, 28, 28)

hidden_dim = 512



class JMVAE_NF_MNIST_CONTOUR(JMVAE_NF):

    shape_mod1, shape_mod2 = (1,28,28), (1,28,28)

    def __init__(self, params):

        # joint_encoder = DoubleHeadMnist(hidden_dim, params.num_hidden_layers,params)
        joint_encoder = DoubleHeadMLP(28 * 28, 28 * 28, hidden_dim, params.latent_dim, 1)
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig
        vae_config = vae_config(input_dim, params.latent_dim)


        # dcca_encoders = load_dcca_mnist_contour()
        # encoder1, encoder2 = TwoStepsEncoder(dcca_encoders[0], params), TwoStepsEncoder(dcca_encoders[1], params)

        encoder1, encoder2 = None, None
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config, encoder=encoder2, decoder=decoder2)

        ])
        super(JMVAE_NF_MNIST_CONTOUR, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_mnist_contour'

        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'contour'
        self.to_tensor = True
        self.classifier1 = None
        self.classifier2 = None

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = None):
        train, test, val = MNIST_CONTOUR_DL(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test, val












