import numpy as np
import torch
import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from bivae.models.nn import Encoder_VAE_SVHN

from ..objectives import cca_loss


class DeepCCA_MNIST_SVHN(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_MNIST_SVHN, self).__init__()
        self.model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), outdim_size))
        self.model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), outdim_size))

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        print('DeepCCA model initialized')
        
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """

        # feature * batch_size
        output1 = self.model1(x1.reshape(x1.shape[0], -1)).embedding
        output2 = self.model2(x2).embedding

        return output1, output2


# def load_dcca_mnist_svhn():

#     model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), 16))
#     model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), 16))

#     model1.load_state_dict(torch.load('../experiments/dcca/mnist_svhn/model1.pt'))
#     model2.load_state_dict(torch.load('../experiments/dcca/mnist_svhn/model2.pt'))

#     return [model1, model2]



class wrapper_encoder_lcca_model1(nn.Module):

    def __init__(self, dim):
        super(wrapper_encoder_lcca_model1, self).__init__()
        # get the outdim size of the encoders from the json file

        model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), 16))
        model1.load_state_dict(torch.load('../experiments/dcca/mnist_svhn/model1.pt'))
        self.latent_dim = dim

        self.encoder = model1
        self.m = np.load('../experiments/dcca/mnist_svhn/l_cca_m.npy')[0]
        self.m = torch.tensor(self.m).cuda().float()
        self.w = np.load('../experiments/dcca/mnist_svhn/l_cca_w.npy')[0]
        self.w = torch.tensor(self.w).cuda().float()
        
    def forward(self, x):
        h = self.encoder(x)['embedding']
        result = h - self.m.reshape([1, -1]).repeat(len(h), 1)
        result = torch.mm(result, self.w)
        o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        return o

class wrapper_encoder_lcca_model2(nn.Module):

    def __init__(self, dim):
        super(wrapper_encoder_lcca_model2, self).__init__()

        model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), 16))
        model2.load_state_dict(torch.load('../experiments/dcca/mnist_svhn/model2.pt'))
        self.latent_dim = dim

        self.encoder = model2
        self.m = np.load('../experiments/dcca/mnist_svhn/l_cca_m.npy')[1]
        self.m = torch.tensor(self.m).cuda().float()
        self.w = np.load('../experiments/dcca/mnist_svhn/l_cca_w.npy')[1]
        self.w = torch.tensor(self.w).cuda().float()

    def forward(self, x):
        self.encoder.eval()
        h = self.encoder(x)['embedding']
        result = h - self.m.reshape([1, -1]).repeat(len(h), 1)
        result = torch.mm(result, self.w)

        o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        return o

def load_dcca_mnist_svhn(dim=16):
    print('loading diminished dcca with additional linear cca')
    model1 = wrapper_encoder_lcca_model1(dim)
    model2 = wrapper_encoder_lcca_model2(dim)



    return [model1, model2]
