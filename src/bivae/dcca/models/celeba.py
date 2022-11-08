import torch
import argparse, json
import torch.nn as nn
from ..objectives import cca_loss
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from pythae.models.nn.default_architectures import Encoder_AE_MLP
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_AE_CELEBA
from pythae.models.base.base_utils import ModelOutput

import numpy as np


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


class wrapper_encoder_lcca_celeb(nn.Module):

    def __init__(self):
        super(wrapper_encoder_lcca_celeb, self).__init__()
        # get the outdim size of the encoders from the json file
        with open('../experiments/dcca/celeba/args.json', 'r') as fcc_file:
            args = argparse.Namespace()
            args.__dict__.update(json.load(fcc_file))
        
        model1 = Encoder_ResNet_AE_CELEBA(VAEConfig((3, 64, 64), args.outdim_size_dcca))
        model1.load_state_dict(torch.load('../experiments/dcca/celeba/model1.pt'))
        self.latent_dim = 40

        self.encoder = model1
        self.m = np.load('../experiments/dcca/celeba/l_cca_m.npy')[0]
        self.w = np.load('../experiments/dcca/celeba/l_cca_w.npy')[0]

    def forward(self, x):

        h = self.encoder(x)['embedding'].cpu().numpy()
        result = h - self.m.reshape([1, -1]).repeat(len(h), axis=0)
        result = np.dot(result, self.w)
        o = ModelOutput(embedding = torch.from_numpy(result).cuda().float())
        return o

class wrapper_encoder_lcca_attributes(nn.Module):

    def __init__(self):
        super(wrapper_encoder_lcca_attributes, self).__init__()
        with open('../experiments/dcca/celeba/args.json', 'r') as fcc_file:
            args = argparse.Namespace()
            args.__dict__.update(json.load(fcc_file))
            
        model2 = Encoder_AE_MLP(VAEConfig((1, 1, 40), args.outdim_size_dcca))
        model2.load_state_dict(torch.load('../experiments/dcca/celeba/model2.pt'))
        self.latent_dim = 40

        self.encoder = model2
        self.m = np.load('../experiments/dcca/celeba/l_cca_m.npy')[1]
        self.w = np.load('../experiments/dcca/celeba/l_cca_w.npy')[1]

    def forward(self, x):

        h = self.encoder(x)['embedding'].cpu().numpy()
        result = h - self.m.reshape([1, -1]).repeat(len(h), axis=0)
        result = np.dot(result, self.w)

        o = ModelOutput(embedding = torch.from_numpy(result).cuda().float())
        return o

def load_dcca_celeba():

    model1 = wrapper_encoder_lcca_celeb()
    model2 = wrapper_encoder_lcca_attributes()



    return [model1, model2]


