# Base JMVAE-NF class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
import wandb

from utils import get_mean, kl_divergence, negative_entropy, update_details
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from pythae.models import my_VAE, VAEConfig
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from utils import extract_rayon
from ..nn import Encoder_VAE_MNIST,Decoder_AE_MNIST

from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP
from ..jmvae_nf import JMVAE_NF

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)
hidden_dim = 512





class JMVAE_NF_CIRCLES(JMVAE_NF):
    def __init__(self, params):
        joint_encoder = DoubleHeadMLP(32*32,32*32,hidden_dim, params.latent_dim, params.num_hidden_layers)
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig
        flow_config = {'n_made_blocks' : 3} if not params.no_nf else {}
        wandb.config.update(flow_config)
        vae_config = vae_config(input_dim, params.latent_dim,**flow_config )

        encoder1, encoder2 = None, None
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
             vae(model_config=vae_config, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config, encoder=encoder2, decoder=decoder2)

        ])
        super(JMVAE_NF_CIRCLES, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_circles_squares'

        self.vaes[0].modelName = 'squares'
        self.vaes[1].modelName = 'circles'

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None):
        # handle merging individual datasets appropriately in sub-class
        # load base datasets
        t1, s1 = CIRCLES.getDataLoaders(batch_size, 'squares', shuffle, device, data_path=self.data_path, transform=transform)
        t2, s2 = CIRCLES.getDataLoaders(batch_size, 'circles', shuffle, device, data_path=self.data_path, transform=transform)

        train_circles_discs = TensorDataset([t1.dataset, t2.dataset])
        test_circles_discs = TensorDataset([s1.dataset, s2.dataset])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train, test



    def analyse_rayons(self,data, r0, r1, runPath, epoch):
        m,s,zxy = self.analyse_joint_posterior(data,n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch))
        wandb.log({'joint_embedding' : wandb.Image('{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch))})
        plot_embeddings_colorbars(zx, zy,r0,r1,'{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch))
        wandb.log({'uni_embedding' : wandb.Image('{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch))})

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        JMVAE_NF.sample_from_conditional(self,data, runPath, epoch,n)
        if epoch == self.max_epochs:
            self.conditional_rdist(data, runPath,epoch)

    def conditional_rdist(self,data,runPath,epoch,n=100):
        bdata = [d[:8] for d in data]
        samples = self._sample_from_conditional(bdata,n)
        samples = torch.cat([torch.stack(samples[0][1]), torch.stack(samples[1][0])], dim=1)
        r = extract_rayon(samples)
        plot_hist(r,'{}/hist_{:03d}.png'.format(runPath, epoch))
        wandb.log({'histograms' : wandb.Image('{}/hist_{:03d}.png'.format(runPath, epoch))})

    def extract_hist_values(self,samples):
        samples = torch.cat([torch.stack(samples[0][1]), torch.stack(samples[1][0])], dim=1)
        return extract_rayon(samples), (0,1), 10

    def compute_metrics(self, data, runPath, epoch, classes=None):
        m = JMVAE_NF.compute_metrics(self, epoch)
        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)
        r, range, bins = self.extract_hist_values(samples)
        sm =  {'neg_entropy' : negative_entropy(r.cpu(), range, bins)}
        update_details(sm,m)
        return sm

