import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss
from bivae.my_pythae.models.nn import Encoder_VAE_MLP, Encoder_VAE_SVHN
from pythae.models.base import BaseAEConfig as vae_config

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class DeepCCA_MNIST_SVHN(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA_MNIST_SVHN, self).__init__()
        self.model1 = Encoder_VAE_MLP(vae_config((1,28,28), outdim_size))
        self.model2 = Encoder_VAE_SVHN(vae_config((3,32,32), outdim_size))

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





# class DeepCCA(nn.Module):
#     def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
#         super(DeepCCA, self).__init__()
#         self.model1 = MlpNet(layer_sizes1, input_size1).double()
#         self.model2 = MlpNet(layer_sizes2, input_size2).double()
#
#         self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
#         print('DeepCCA model initialized')
#     def forward(self, x1, x2):
#         """
#
#         x1, x2 are the vectors needs to be make correlated
#         dim=[batch_size, feats]
#
#         """
#
#         # feature * batch_size
#         output1 = self.model1(x1)
#         output2 = self.model2(x2)
#
#         return output1, output2
