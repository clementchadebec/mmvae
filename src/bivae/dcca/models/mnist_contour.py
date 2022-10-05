import torch
import torch.nn as nn
from ..objectives import cca_loss
from bivae.my_pythae.models.nn import Encoder_VAE_MLP, Encoder_VAE_SVHN
from bivae.my_pythae.models.vae import VAEConfig



class DeepCCA_MNIST_CONTOUR(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_MNIST_CONTOUR, self).__init__()
        self.model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), outdim_size))
        self.model2 = Encoder_VAE_MLP(VAEConfig((1,28,28), outdim_size))

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        print('DeepCCA model initialized')
    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """

        # feature * batch_size
        output1 = self.model1(x1.reshape(x1.shape[0], -1)).embedding
        output2 = self.model2(x2.reshape(x2.shape[0], -1)).embedding

        return output1, output2


def load_dcca_mnist_contour():

    model1 = Encoder_VAE_MLP(VAEConfig((1,28,28), 15))
    model2 = Encoder_VAE_MLP(VAEConfig((1,28,28), 15))

    model1.load_state_dict(torch.load('../dcca/mnist_contour/model1.pt'))
    model2.load_state_dict(torch.load('../dcca/mnist_contour/model2.pt'))

    model1.eval()
    model2.eval()

    model1.cuda()
    model2.cuda()

    return [model1, model2]


