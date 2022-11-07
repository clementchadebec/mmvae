# MMVAE specification for MNIST-SVHN-FASHION experiment

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import wandb
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from torchvision import transforms

from bivae.analysis.classifiers import load_pretrained_mnist, load_pretrained_svhn
from bivae.dataloaders import MNIST_SVHN_FASHION_DL, BINARY_MNIST_SVHN_DL
from bivae.my_pythae.models import my_VAE, laplace_VAE
from bivae.utils import update_details
from bivae.vis import plot_hist
from .mmvae import MMVAE
from ..nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.utils import add_channels, unpack_data
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader
from bivae.analysis.accuracies import compute_accuracies
from bivae.dataloaders import BasicDataset

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}


class MNIST_SVHN_FASHION(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae_config3 = vae_config((1,28,28), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE


        e1, e2,e3 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2), Encoder_VAE_MLP(vae_config3)
        d1, d2,d3 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2), Decoder_AE_MLP(vae_config3)

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=e1, decoder=d1),
            vae(model_config=vae_config2, encoder=e2, decoder=d2),
            vae(vae_config3,e3,d3)

        ])
        super(MNIST_SVHN_FASHION, self).__init__(params, vaes)
        self.modelName = 'mmvae_msf'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.vaes[2].modelName = 'fashion'
        # self.lik_scaling = ((3*32*32)/(1*28*28),1,(3*32*32)/(1*28*28))
        self.lik_scaling = (1,1,1)
        wandb.log({'lik_scaling' : self.lik_scaling})

    def set_classifiers(self):
        
        self.classifier1 = load_pretrained_mnist()
        self.classifier2 = load_pretrained_svhn()
        


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def conditional_labels(self, data, n_data=8, ns=30):
        """ Sample ns from the conditional distribution (for each of the first n_data)
        and compute the labels in this conditional distribution (based on the
        predefined classifiers)"""

        bdata = [d[:n_data] for d in data]
        samples = self._sample_from_conditional( bdata, n=ns)
        labels = [[None for _ in range(self.mod)] for _ in range(self.mod) ]
        for i in range(self.mod):
            for j in range(self.mod):
                if i!=j:
                    preds = self.classifiers[j](samples[i][j].permute(1,0,2,3,4).resize(n_data*ns, *self.shape_mods[j]))
                    labels[i][j] = torch.argmax(preds, dim=1).reshape(n_data, ns)
        return labels
    
    

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""
        
        self.set_classifiers()
        general_metrics = MMVAE.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies

    def compute_fid(self, batch_size):
        
        #TODO : Check that this function is working

        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = self.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[] for i in range(self.mod)]

        for dataT in test:
            data = unpack_data(dataT)
            for i in range(self.mod):
                ref_activations[i].append(model(data[i]))
            

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = self.getDataLoaders(batch_size)

        gen_samples = [[[] for j in range(self.mod)] for i in range(self.mod)]
        for dataT in test:
            data = unpack_data(dataT)
            gen = self._sample_from_conditional(data, n=1)
            for i in range(self.mod):
                for j in range(self.mod):
                    gen_samples[i][j].extend(gen[i][j])
            

        gen_samples = [[torch.cat(g).squeeze(0) for g in row] for row in gen_samples]
        print(gen_samples[0].shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_activations = [[[] for j in range(self.mod)] for i in range( self.mod)]
        
        for i in range(self.mod):
            for j in range(self.mod):
                if i != j :
                    gen = torch.cat(gen_samples[i][j]).squeeze(0)
                    dataset = BasicDataset(gen,tx)
                    dl = DataLoader(dataset, batch_size)
                    # Compute all the activations
                    for data in dl:
                        gen_activations[i][j].append(model[data])
                


        cond_fids = {}
        
        for i in range(self.mod): # modality sampled
            mu_ref = np.mean(ref_activations[i], axis=0)
            sigma_ref = np.cov(ref_activations[i],rowvar=False )
            for j in range(self.mod): # modality we condition on for sampling
                if i != j:
                    # Compute mean and sigma
                    mu_gen = np.mean(np.concatenate(gen_activations[j][i]))
                    sigma_gen = np.cov(np.concatenate(gen[j][i]), rowvar=False)


                    cond_fids[f'fid_{j}_{i}'] = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

        return cond_fids













