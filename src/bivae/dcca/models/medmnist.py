import torch
import torch.nn as nn
from ..objectives import cca_loss
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF
from bivae.models.nn.medmnist import Encoder_ResNet_AE_medmnist, Encoder_Resnet18_AE_medmnist


encoder = Encoder_ResNet_AE_medmnist


class DeepCCA_MedMNIST(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_MedMNIST, self).__init__()

        
        self.model1 = encoder(VAEConfig((1,28,28), outdim_size))
        self.model2 = encoder(VAEConfig((3,28,28), outdim_size))
        

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        print('DeepCCA model initialized')
        
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be made correlated
        dim=[batch_size, feats]

        """

        # feature * batch_size
        output1 = self.model1(x1).embedding
        output2 = self.model2(x2).embedding

        return output1, output2


def load_dcca_medmnist(outdim_size):

    model1 = encoder(VAEConfig((1,28,28), outdim_size))
    model2 = encoder(VAEConfig((3,28,28), outdim_size))

    model1.load_state_dict(torch.load('../experiments/dcca/medmnist/model1.pt'))
    model2.load_state_dict(torch.load('../experiments/dcca/medmnist/model2.pt'))

    return [model1, model2]


