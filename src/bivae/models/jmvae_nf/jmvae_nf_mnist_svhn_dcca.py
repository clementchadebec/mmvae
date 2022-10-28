# JMVAE_NF specification for MNIST-SVHN experiment --> Using DCCA to extract shared information

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from torchvision import transforms

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn
from torchvision.utils import save_image
import pythae
from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from bivae.models.nn import Encoder_VAE_SVHN
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from bivae.utils import extract_rayon
from bivae.dataloaders import MNIST_SVHN_DL, MultimodalBasicDataset
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Decoder_VAE_SVHN, TwoStepsDecoder, TwoStepsEncoder
import torch.nn.functional as F

from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies
from bivae.analysis.pytorch_fid import calculate_frechet_distance, wrapper_inception
from bivae.utils import unpack_data, add_channels
from bivae.dcca.models import load_dcca_mnist_svhn

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

# Define the classifiers for analysis


class JMVAE_NF_DCCA_MNIST_SVHN(JMVAE_NF):

    shape_mod1, shape_mod2 = (1, 28, 28), (3, 32, 32)
    modelName = 'jmvae_nf_dcca_mnist_svhn'


    def __init__(self, params):

        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig

        # Define the joint encoder
        hidden_dim = 512
        pre_configs = [VAEConfig((1, 28, 28), 20), VAEConfig((3, 32, 32), 20)]
        joint_encoder = DoubleHeadJoint(hidden_dim,pre_configs[0], pre_configs[1],
                                        Encoder_VAE_MLP ,
                                        Encoder_VAE_SVHN,
                                        params)

        # Define the unimodal encoders config
        vae_config1 = vae_config((1, 28, 28), params.latent_dim)
        vae_config2 = vae_config((3, 32, 32), params.latent_dim)

        if self.params.dcca :
            # First load the DCCA encoders
            self.dcca = load_dcca_mnist_svhn()

            # Then add the flows
            encoder1 = TwoStepsEncoder(self.dcca[0], params)
            encoder2 = TwoStepsEncoder(self.dcca[1], params)
        else :
            encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2)


        # Define the decoders
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        

        # Then define the vaes
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)
        ])

        super(JMVAE_NF_DCCA_MNIST_SVHN, self).__init__(params, joint_encoder, vaes)

        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1) if params.llik_scaling == 0.0 else (
        params.llik_scaling, 1)
        self.to_tensor = True

    
    def set_classifiers(self):

        self.classifier1,self.classifier2 = load_pretrained_mnist(), load_pretrained_svhn()
        return 


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""


        general_metrics = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,self.classifier1,self.classifier2,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies

    def compute_recon_loss(self,x,recon,m):
        """Change the way we compute the reocnstruction, through the filter of DCCA"""
        t = self.dcca[m](x).embedding
        recon_t = self.dcca[m](recon).embedding
        return F.mse_loss(t,recon_t,reduction='sum')


    def compute_fid(self, batch_size):

        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = self.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[],[]]

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations[0].append(model(data[0]))
            ref_activations[1].append(model(data[1]))

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = self.getDataLoaders(batch_size)

        gen_samples = [[],[]]
        for dataT in test:
            data = unpack_data(dataT)
            gen = self._sample_from_conditional(data, n=1)
            gen_samples[0].extend(gen[1][0])
            gen_samples[1].extend(gen[0][1])

        gen_samples = [torch.cat(g).squeeze(0) for g in gen_samples]
        print(gen_samples[0].shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_dataset = MultimodalBasicDataset(gen_samples, tx)
        gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size)

        gen_activations = [[],[]]
        for dataT in gen_dataloader:
            data = unpack_data(dataT)
            gen_activations[0].append(model(data[0]))
            gen_activations[1].append(model(data[1]))
        gen_activations = [np.concatenate(g) for g in gen_activations]

        cond_fids = {}
        for i in range(len(ref_activations)):
            mu1, mu2 = np.mean(ref_activations[i], axis=0), np.mean(gen_activations[i], axis=0)
            sigma1, sigma2 = np.cov(ref_activations[i], rowvar=False), np.cov(gen_activations[i], rowvar=False)

            # print(mu1.shape, sigma1.shape)

            cond_fids[f'fid_{i}'] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return cond_fids













