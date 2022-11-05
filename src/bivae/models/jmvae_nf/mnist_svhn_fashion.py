# JMVAE_NF specification for MNIST-SVHN experiment --> Using DCCA to extract shared information

from itertools import combinations
from re import M

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
from bivae.dataloaders import MNIST_SVHN_FASHION_DL, MultimodalBasicDataset, BasicDataset
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Decoder_VAE_SVHN, TwoStepsDecoder, TwoStepsEncoder
import torch.nn.functional as F

from ..nn import DoubleHeadMLP, MultipleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies
from bivae.analysis.pytorch_fid import calculate_frechet_distance, wrapper_inception
from bivae.utils import unpack_data, add_channels
from bivae.dcca.models.mnist_svhn_fashion import load_dcca_mnist_svhn_fashion

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

# Define the classifiers for analysis


class MNIST_SVHN_FASHION(JMVAE_NF):

    shape_mods = [(1, 28, 28), (3, 32, 32), (1,28,28)]
    modelName = 'jmvae_nf_mnist_svhn_fashion'


    def __init__(self, params):

        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig

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
        vae = my_VAE_IAF if not params.no_nf else my_VAE
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

        self.classifier1,self.classifier2 = load_pretrained_mnist(), load_pretrained_svhn()
        return 


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""


        general_metrics = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self,data,classes,n_data,ns)

        update_details(accuracies, general_metrics)
        return accuracies

    def compute_recon_loss(self,x,recon,m):
        """Change the way we compute the reocnstruction, through the filter of DCCA"""
        t = self.dcca[m](x).embedding
        recon_t = self.dcca[m](recon).embedding
        return F.mse_loss(t,recon_t,reduction='sum')


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













