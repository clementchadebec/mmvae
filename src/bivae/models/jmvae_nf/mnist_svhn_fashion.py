# JMVAE_NF specification for MNIST-SVHN experiment --> Using DCCA to extract shared information


import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from pythae.models import VAE_IAF_Config, VAEConfig
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

from bivae.analysis import (compute_accuracies, load_pretrained_fashion,
                            load_pretrained_mnist, load_pretrained_svhn)
from bivae.dataloaders import MNIST_SVHN_FASHION_DL
from bivae.dcca.models.mnist_svhn_fashion import load_dcca_mnist_svhn_fashion
from bivae.models.nn import Encoder_VAE_SVHN
from bivae.my_pythae.models import (VAE_MAF_Config, my_VAE, my_VAE_IAF,
                                    my_VAE_MAF)
from bivae.utils import add_channels, update_details

from ..jmvae_nf import JMVAE_NF
from ..modalities.trimodal import *
from ..nn import Decoder_VAE_SVHN, MultipleHeadJoint, TwoStepsEncoder
import time

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

# Define the classifiers for analysis


class MNIST_SVHN_FASHION(JMVAE_NF):

    shape_mods = [(1, 28, 28), (3, 32, 32), (1,28,28)]
    modelName = 'jmvae_nf_mnist_svhn_fashion'


    def __init__(self, params):

        if params.no_nf :
            print('No normalizing flows')
            vae_config, vae = VAEConfig , my_VAE
        else :
            vae_config = VAE_IAF_Config if params.flow == 'iaf' else VAE_MAF_Config
            vae = my_VAE_IAF if params.flow == 'iaf' else my_VAE_MAF

            
        # Define the joint encoder
        hidden_dim = 512
        pre_configs = [VAEConfig((1, 28, 28), 20), VAEConfig((3, 32, 32), 20), VAEConfig((1,28,28),20)]
        joint_encoder = MultipleHeadJoint(hidden_dim,pre_configs,
                                        [Encoder_VAE_MLP , Encoder_VAE_SVHN, Encoder_VAE_MLP],
                                        params)

        # Define the unimodal encoders config
        vae_config1 = vae_config((1, 28, 28), params.latent_dim)
        vae_config2 = vae_config((3, 32, 32), params.latent_dim)
        vae_config3 = vae_config((1,28,28), params.latent_dim)

        if params.dcca :
            # First load the DCCA encoders
            self.dcca = load_dcca_mnist_svhn_fashion()

            # Then add the flows
            e1 = TwoStepsEncoder(self.dcca[0], params)
            e2 = TwoStepsEncoder(self.dcca[1], params)
            e3 = TwoStepsEncoder(self.dcca[2], params)
        else :
            e1,e2,e3 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2), Encoder_VAE_MLP(vae_config3)


        # Define the decoders
        d1,d2,d3 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2), Decoder_AE_MLP(vae_config3)
        

        # Then define the vaes

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=e1, decoder=d1),
            vae(model_config=vae_config2, encoder=e2, decoder=d2),
            vae(vae_config3,e3,d3)
        ])

        super(MNIST_SVHN_FASHION, self).__init__(params, joint_encoder, vaes)

        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.vaes[2].modelName = 'fashion'
        self.lik_scaling = (1,1,1)

    
    def set_classifiers(self):

        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn(), load_pretrained_fashion()]
        return 


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        self.set_classifiers()
        # general_metrics = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)
        general_metrics = {}
        update_details(accuracies, general_metrics)
        update_details(accuracies, compute_poe_subset_accuracy(self,data,classes,n_data,ns))
        return accuracies




    def compute_fid(self, batch_size):
        return fid(self, batch_size)

    def analyse(self, data, runPath, epoch, classes):
        # TODO
        pass
    
    def analyse_posterior(self, data, n_samples, runPath, epoch, ticks, N):
        pass
    
    def compute_conditional_likelihoods(self, data, K=1000, batch_size_K=100):
        d =  super().compute_conditional_likelihoods(data, K, batch_size_K)
        poe_ll = compute_all_cond_ll_from_poe_subsets(self,data,K,batch_size_K=500)
        update_details(d,poe_ll)
        
        
        return d
    
    
    def sample_from_poe(self, data, runPath, epoch, n=10, divide_prior=False):
        print("passing through sample_from_poe", divide_prior)
        sample_from_poe_vis(self, data, runPath,epoch, n, divide_prior=divide_prior)
    
    
    