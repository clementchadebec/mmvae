''' A joint multimodal autoencoder for mnist-fashion dataset'''

from torch import nn
import torch.nn.functional as F
from utils import Constants
import torch
from .nn.joint_encoders import BaseEncoder

from .mmvae_mnist_fashion import MNIST_FASHION
from .jmvae import JMVAE

# Constants
size1, size2 = 28*28 , 28*28 # modality 1 size, modality 2 size
hidden_dim = 512


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))




class J_MNIST_FASHION(MNIST_FASHION, JMVAE):

    def __init__(self, params):
        MNIST_FASHION.__init__(self,params)
        joint_encoder = BaseEncoder(size1, size2, hidden_dim, params.latent_dim, params.num_hidden_layers)
        JMVAE.__init__(self,params, joint_encoder, self.vaes)

    def analyse_posterior(self,data, n_samples,runPath,epoch, ticks=None):
        means, stds = MNIST_FASHION.analyse_posterior(self,data, n_samples, runPath, epoch, ticks)
        m, s, _ = JMVAE.analyse_joint_posterior(self,data, n_samples)
        means.append(m)
        stds.append(s)
