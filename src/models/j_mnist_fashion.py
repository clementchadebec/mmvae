''' A joint multimodal autoencoder for mnist-fashion dataset'''

from torch import nn
import torch.nn.functional as F
from utils import Constants
import torch

from .mmvae_mnist_fashion import MNIST_FASHION
from .jmvae import JMVAE

# Constants
joint_input_size = 28*28 + 28*28 # modality 1 size, modality 2 size
hidden_dim = 400


def extra_hidden_layer():
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


# Classes
class Enc(nn.Module):
    """ Simple MLP as a joint encoder """

    def __init__(self, latent_dim, num_hidden_layers=1):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(joint_input_size, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer() for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        print(x.size())
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta


class J_MNIST_FASHION(MNIST_FASHION, JMVAE):

    def __init__(self, params):
        MNIST_FASHION.__init__(self,params)
        JMVAE.__init__(self,params, Enc(params.latent_dim, params.num_hidden_layers), self.vaes)
