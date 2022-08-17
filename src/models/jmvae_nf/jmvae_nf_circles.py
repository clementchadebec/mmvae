# Base JMVAE-NF class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
import wandb
from numpy.random import randint
import numpy as np
from utils import get_mean, kl_divergence, negative_entropy, update_details
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from pythae.models import my_VAE, VAEConfig
from torchnet.dataset import TensorDataset, ResampleDataset
from torch.utils.data import DataLoader
from utils import extract_rayon
from ..nn import Encoder_VAE_SVHN,Decoder_VAE_SVHN

from dataloaders import CIRCLES_SQUARES_DL
from ..nn import DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from analysis import CirclesClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)
hidden_dim = 512

classifier1, classifier2 = CirclesClassifier(), CirclesClassifier()
path1 = '../experiments/classifier_squares/2022-06-28/model_4.pt'
path2 = '../experiments/classifier_circles/2022-06-28/model_4.pt'
classifier1.load_state_dict(torch.load(path1))
classifier2.load_state_dict(torch.load(path2))
# Set in eval mode
classifier1.eval()
classifier2.eval()
# Set to cuda
classifier1.cuda()
classifier2.cuda()

class JMVAE_NF_CIRCLES(JMVAE_NF):
    def __init__(self, params):
        params.input_dim = input_dim

        joint_encoder = DoubleHeadJoint(hidden_dim, params,params,Encoder_VAE_SVHN,Encoder_VAE_SVHN, params)
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig
        flow_config = {'n_made_blocks' : 2} if not params.no_nf else {}
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
        self.to_tensor = False

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None):
        # handle merging individual datasets appropriately in sub-class
        # load base datasets
        dl = CIRCLES_SQUARES_DL(self.data_path)
        train, test, val = dl.getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def analyse_rayons(self,data, r0, r1, runPath, epoch, filters):
        m,s,zxy = self.analyse_joint_posterior(data,n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch), filters)
        wandb.log({'joint_embedding' : wandb.Image('{}/embedding_rayon_joint{:03}.png'.format(runPath,epoch))})
        plot_embeddings_colorbars(zx, zy,r0,r1,'{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch), filters)
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

    def compute_metrics(self, data, runPath, epoch, classes=None,freq=10):
        m = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)

        # Compute cross accuracy of generation
        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)

        preds1 = classifier2(torch.stack(samples[0][1]))
        preds0 = classifier1(torch.stack(samples[1][0]))

        labels0 = torch.argmax(preds0, dim=-1).reshape(100, 100)
        labels1 = torch.argmax(preds1, dim=-1).reshape(100, 100)
        classes_mul = torch.stack([classes[0][:100] for _ in np.arange(100)]).cuda()
        acc1 = torch.sum(labels1 == classes_mul)/(100*100)
        acc0 = torch.sum(labels0 == classes_mul)/(100*100)

        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)
        r, range, bins = self.extract_hist_values(samples)
        sm =  {'neg_entropy' : negative_entropy(r.cpu(), range, bins), 'acc0' : acc0, 'acc1' : acc1}
        update_details(sm,m)

        print('Eval metrics : ', sm)
        return sm

