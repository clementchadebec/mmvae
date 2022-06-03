# Base JMVAE-NF class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist

from utils import get_mean, kl_divergence
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from utils import extract_rayon

from ..vae_circles import CIRCLES
from ..joint_encoders import DoubleHeadMLP
from ..jmvae_nf import JMVAE_NF

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)
hidden_dim = 512

vae = my_VAE_IAF
vae_config = VAE_IAF_Config


class JMVAE_NF_CIRCLES(JMVAE_NF):
    def __init__(self, params):
        joint_encoder = DoubleHeadMLP(32*32,32*32,hidden_dim, params.latent_dim, params.num_hidden_layers)
        vae = my_VAE_IAF
        vae_config = VAE_IAF_Config(input_dim, params.latent_dim)
        super(JMVAE_NF_CIRCLES, self).__init__(params, joint_encoder, vae, vae_config)
        self.modelName = 'jmvae_nf_circles_squares'



    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        # load base datasets
        t1, s1 = CIRCLES.getDataLoaders(batch_size, 'squares', shuffle, device, data_path=self.data_path)
        t2, s2 = CIRCLES.getDataLoaders(batch_size, 'circles', shuffle, device, data_path=self.data_path)

        train_circles_discs = TensorDataset([t1.dataset, t2.dataset])
        test_circles_discs = TensorDataset([s1.dataset, s2.dataset])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train, test



    def analyse_rayons(self,data, r0, r1, runPath, epoch):
        m,s,zxy = self.analyse_joint_posterior(data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_{:03}.png'.format(runPath,epoch))

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        JMVAE_NF.sample_from_conditional(self,data, runPath, epoch,n)
        self.conditional_rdist(data, runPath,epoch)

    def conditional_rdist(self,data,runPath,epoch,n=30):
        bdata = [d[:8] for d in data]
        samples = self._sample_from_conditional(bdata,n)
        samples = torch.cat([torch.stack(samples[0][1]), torch.stack(samples[1][0])], dim=1)
        r = extract_rayon(samples)
        plot_hist(r,'{}/hist_{:03d}.png'.format(runPath, epoch))



