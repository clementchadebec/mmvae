

import torch.nn as nn
from pythae.models import VAE_IAF_Config, VAEConfig
from pythae.models.nn.default_architectures import (Decoder_AE_MLP,
                                                    Encoder_VAE_MLP)
from torchvision import transforms

from bivae.dcca.models import load_dcca_mnist_svhn
from bivae.models.modalities.mnist_svhn import fid
from bivae.my_pythae.models import my_VAE, my_VAE_IAF
from bivae.my_pythae.models.vae_maf import VAE_MAF_Config, my_VAE_MAF
from bivae.models.nn.medmnist import Encoder_ResNet_VAE_medmnist, Decoder_ResNet_AE_medmnist
from bivae.dataloaders import PATH_BLOOD_DL
from bivae.analysis.classifiers import ClassifierBLOOD, ClassifierPATH

from ..nn import Decoder_VAE_SVHN, TwoStepsEncoder
from bivae.analysis.accuracies import compute_accuracies
from bivae.utils import update_details
from bivae.dcca.models import load_dcca_medmnist


class medmnist_utils():
    
    
    def __init__(self, params) -> None:
        
        self.data_path = params.data_path
        self.shape_mods = [(3,28,28), (1,28,28)]
        self.lik_scaling = (1,1)
        
    def get_vaes(self,params):
        
        # Define the VAEs
        if params.no_nf :
            vae_config, vae = VAEConfig , my_VAE
        else :
            vae_config = VAE_IAF_Config if params.flow == 'iaf' else VAE_MAF_Config
            vae = my_VAE_IAF if params.flow == 'iaf' else my_VAE_MAF
            
        

        # Define the unimodal encoders config
        vae_config1 = vae_config((3, 28, 28), params.latent_dim)
        vae_config2 = vae_config((3, 28, 28), params.latent_dim)

        if params.dcca :
            # First load the DCCA encoders
            self.dcca = load_dcca_medmnist()

            # Then add the flows
            encoder1 = TwoStepsEncoder(self.dcca[0], params)
            encoder2 = TwoStepsEncoder(self.dcca[1], params)
        else :
            encoder1, encoder2 = Encoder_ResNet_VAE_medmnist(vae_config1), Encoder_ResNet_VAE_medmnist(vae_config2)


        # Define the decoders
        decoder1, decoder2 = Decoder_ResNet_AE_medmnist(vae_config1), Decoder_ResNet_AE_medmnist(vae_config2)
        

        # Then define the vaes
        
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)
        ])
        
        vaes[0].modelName = 'PATH'
        vaes[1].modelName = 'BLOOD'
        
        return vaes
    
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        
        train, test, val = PATH_BLOOD_DL().getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val
    
    def set_classifiers(self):

        self.classifiers = [ClassifierPATH(), ClassifierBLOOD()]
        return 
    
    def compute_fid(self, batch_size):
        return fid(self,batch_size)
    
    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""
        
        self.set_classifiers()
        general_metrics = super().compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies