import torch
import torch.nn as nn
from ..objectives import cca_loss
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from pythae.models.nn.default_architectures import Encoder_AE_MLP
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_AE_CELEBA



class DeepCCA_celeba(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_celeba, self).__init__()
        self.model1 = Encoder_ResNet_AE_CELEBA(VAEConfig((3,64,64), outdim_size))
        self.model2 = Encoder_AE_MLP(VAEConfig((1,1,40), outdim_size))

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        print('DeepCCA model initialized')
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """

        # feature * batch_size
        output1 = self.model1(x1).embedding
        output2 = self.model2(x2).embedding

        return output1, output2


def load_dcca_celeba():

    model1 = Encoder_ResNet_AE_CELEBA(VAEConfig((3,64,64), 40))
    model2 = Encoder_AE_MLP(VAEConfig((1,1,40), 40))

    model1.load_state_dict(torch.load('../dcca/celeba/model1.pt'))
    model2.load_state_dict(torch.load('../dcca/celeba/model2.pt'))

    return [model1, model2]


