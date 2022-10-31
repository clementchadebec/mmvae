import torch
import torch.nn as nn
from ..objectives import mcca_loss
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from bivae.models.nn import Encoder_VAE_SVHN



class DeepCCA_MNIST_SVHN_FASHION(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_MNIST_SVHN_FASHION, self).__init__()
        self.model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), outdim_size))
        self.model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), outdim_size))
        self.model3 = Encoder_VAE_MLP(VAEConfig(input_dim=(1,28,28), latent_dim=outdim_size))

        self.loss = mcca_loss(outdim_size, use_all_singular_values, device).loss
        print('DeepCCA model initialized')
        
    def forward(self, x_list):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """

        # feature * batch_size
        output1 = self.model1(x_list[0].reshape(x_list[0].shape[0], -1)).embedding
        output2 = self.model2(x_list[1]).embedding
        output3 = self.model3(x_list[2].reshape(len(x_list[2]), -1)).embedding

        return [output1, output2, output3]


def load_dcca_mnist_svhn_fashion():

    model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), 16))
    model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), 16))
    model3 = Encoder_VAE_MLP(VAEConfig((1,28,28), 16))

    model1.load_state_dict(torch.load('../dcca/mnist_svhnmodel1.pt'))
    model2.load_state_dict(torch.load('../dcca/mnist_svhnmodel2.pt'))

    return [model1, model2, model3]


