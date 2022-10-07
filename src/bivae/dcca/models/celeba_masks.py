import torch
import torch.nn as nn
from ..objectives import cca_loss
from pythae.models.nn.benchmarks.celeba import Encoder_AE_CELEBA
from pythae.models.vae.vae_config import VAEConfig


class DeepCCA_CelebA_Masks(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_CelebA_Masks, self).__init__()
        self.model1 = Encoder_AE_CELEBA(VAEConfig((3,64,64), outdim_size))
        self.model2 = Encoder_AE_CELEBA(VAEConfig((3,64,64), outdim_size))

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


def load_dcca_mnist_svhn():

    model1 = Encoder_AE_CELEBA(VAEConfig((3,64,64), 15)) # ! Change the outdim size !
    model2 = Encoder_AE_CELEBA(VAEConfig((3,64,64), 15)) # ! Change the outdim size !

    model1.load_state_dict(torch.load('../dcca/celeba_masks/model_celeba.pt'))
    model2.load_state_dict(torch.load('../dcca/celeba_masks/model_masks.pt'))

    return [model1, model2]


