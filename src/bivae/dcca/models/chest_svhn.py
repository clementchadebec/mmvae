import numpy as np
import torch
import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput

from bivae.models.nn.medmnist import (Encoder_ResNet_AE_medmnist)
from bivae.models.nn import Encoder_VAE_SVHN

from ..objectives import cca_loss



class DCCA_CHEST_SVHN(nn.Module):

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DCCA_CHEST_SVHN, self).__init__()

        
        self.model1 = Encoder_ResNet_AE_medmnist(VAEConfig((1,28,28), outdim_size))
        self.model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), outdim_size))
        

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
    
    

class wrapper_encoder_lcca_model1(nn.Module):

    def __init__(self, dim):
        super(wrapper_encoder_lcca_model1, self).__init__()
        # get the outdim size of the encoders from the json file

        model1 = Encoder_ResNet_AE_medmnist(VAEConfig((1,28,28), 3))
        model1.load_state_dict(torch.load('../experiments/dcca/chest_svhn/model1.pt'))
        self.latent_dim = dim

        self.encoder = model1
        self.m = np.load('../experiments/dcca/chest_svhn/l_cca_m.npy')[0]
        self.m = torch.tensor(self.m).cuda().float()
        self.w = np.load('../experiments/dcca/chest_svhn/l_cca_w.npy')[0]
        self.w = torch.tensor(self.w).cuda().float()
        
    def forward(self, x):
        h = self.encoder(x)['embedding']
        result = h - self.m.reshape([1, -1]).repeat(len(h), 1)
        result = torch.mm(result, self.w)
        # o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        o = ModelOutput(embedding = h)
        
        return o

class wrapper_encoder_lcca_model2(nn.Module):

    def __init__(self, dim):
        super(wrapper_encoder_lcca_model2, self).__init__()

        model2 = Encoder_VAE_SVHN(VAEConfig((3,32,32), 3))
        model2.load_state_dict(torch.load('../experiments/dcca/chest_svhn/model2.pt'))
        self.latent_dim = dim

        self.encoder = model2
        self.m = np.load('../experiments/dcca/chest_svhn/l_cca_m.npy')[1]
        self.m = torch.tensor(self.m).cuda().float()
        self.w = np.load('../experiments/dcca/chest_svhn/l_cca_w.npy')[1]
        self.w = torch.tensor(self.w).cuda().float()

    def forward(self, x):
        self.encoder.eval()
        h = self.encoder(x)['embedding']
        result = h - self.m.reshape([1, -1]).repeat(len(h), 1)
        result = torch.mm(result, self.w)

        # o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        o = ModelOutput(embedding = h)

        return o



def load_dcca_chest_svhn(outdim_size):

    model1 = wrapper_encoder_lcca_model1(outdim_size)
    model2 = wrapper_encoder_lcca_model2(outdim_size)

    return [model1, model2]

