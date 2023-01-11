

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
from ..modalities.chest_svhn import chest_svhn_utils

from ..jmvae_nf import JMVAE_NF
from ..nn import Decoder_VAE_SVHN, TwoStepsEncoder
from bivae.models.nn.medmnist import Encoder_ResNet_VAE_medmnist
from bivae.models.nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN



class CHEST_SVHN(JMVAE_NF,chest_svhn_utils):

    modelName = 'jnf_chest_svhn'


    def __init__(self, params):
        
        # Define the joint encoder
        hidden_dim = 512
        pre_configs = [VAEConfig((1, 28, 28), 20), VAEConfig((3, 32, 32), 20)]
        joint_encoder = DoubleHeadJoint(hidden_dim, *pre_configs,Encoder_ResNet_VAE_medmnist, Encoder_VAE_SVHN,params)
        
        chest_svhn_utils.__init__(self,params)
        vaes = self.get_vaes(params)

        JMVAE_NF.__init__(self,params, joint_encoder, vaes)

        
       

    
    

    


    