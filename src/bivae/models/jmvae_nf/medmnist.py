

import torch.distributions as dist
import torch.nn as nn
from pythae.models import VAE_IAF_Config, VAEConfig
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

from bivae.analysis import (compute_accuracies, load_pretrained_mnist,
                            load_pretrained_svhn)
from bivae.dataloaders import MNIST_SVHN_DL
from bivae.dcca.models import load_dcca_mnist_svhn
from bivae.models.modalities.mnist_svhn import fid

from bivae.models.nn.joint_encoders import DoubleHeadJoint
from bivae.my_pythae.models import my_VAE, my_VAE_IAF
from bivae.my_pythae.models.vae_maf import VAE_MAF_Config, my_VAE_MAF
from bivae.utils import add_channels, update_details
from ..modalities.medmnist import medmnist_utils

from ..jmvae_nf import JMVAE_NF
from ..nn import Decoder_VAE_SVHN, TwoStepsEncoder
from bivae.models.nn.medmnist import Encoder_ResNet_VAE_medmnist



class MEDMNIST(JMVAE_NF,medmnist_utils):

    modelName = 'jnf_medmnist'


    def __init__(self, params):
        
        # Define the joint encoder
        hidden_dim = 512
        pre_configs = [VAEConfig((3, 28, 28), 20), VAEConfig((3, 28, 28), 20)]
        joint_encoder = DoubleHeadJoint(hidden_dim, *pre_configs,Encoder_ResNet_VAE_medmnist, Encoder_ResNet_VAE_medmnist,params)
        
        medmnist_utils.__init__(self,params)
        vaes = self.get_vaes(params)

        JMVAE_NF.__init__(self,params, joint_encoder, vaes)

        
       

    
    

    


    