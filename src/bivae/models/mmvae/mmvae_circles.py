# MMVAE CIRCLES SQUARES SPECIFICATION

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
import wandb
import numpy as np

from bivae.utils import get_mean, kl_divergence, negative_entropy, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from bivae.my_pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from bivae.my_pythae.models import my_VAE, VAEConfig
from torchnet.dataset import TensorDataset, ResampleDataset
from torch.utils.data import DataLoader
from bivae.utils import extract_rayon
from ..nn import Encoder_VAE_MNIST,Decoder_AE_MNIST
from bivae.dataloaders import CIRCLES_SQUARES_DL
from bivae.analysis import CirclesClassifier

from ..vae_circles import CIRCLES
from .mmvae import MMVAE

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



class MMVAE_CIRCLES(MMVAE):
    def __init__(self, params):

        vae = my_VAE
        vae_config =  VAEConfig
        flow_config = {}
        wandb.config.update(flow_config)
        vae_config = vae_config(input_dim, params.latent_dim,**flow_config )

        encoder1, encoder2 = None, None
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
             vae(model_config=vae_config, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config, encoder=encoder2, decoder=decoder2)

        ])
        super(MMVAE_CIRCLES, self).__init__(params, vaes)
        self.modelName = 'mmvae_circles_squares'

        self.vaes[0].modelName = 'squares'
        self.vaes[1].modelName = 'circles'

        self.params = params
        self.lik_scaling = (1,1)
        self.to_tensor = False


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None, random=False):
        dl = CIRCLES_SQUARES_DL(self.params.data_path)
        return dl.getDataLoaders(batch_size, shuffle, device, transform)


    def analyse_rayons(self,data, r0, r1, runPath, epoch, filters=[None,None]):
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))

        plot_embeddings_colorbars(zx, zy,r0,r1,'{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch), ax_lim=None, filters=filters)
        wandb.log({'uni_embedding' : wandb.Image('{}/embedding_rayon_uni{:03}.png'.format(runPath,epoch))})

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        MMVAE.sample_from_conditional(self,data, runPath, epoch,n)
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

    def compute_metrics(self, data, runPath, epoch, classes=None, freq=10):
        m = MMVAE.compute_metrics(self, runPath, epoch, freq=freq)
        m = {}
        bdata = [d[:100] for d in data]
        samples = self._sample_from_conditional(bdata, n=100)

        # Compute conditional accuracy
        preds1 = classifier2(torch.stack(samples[0][1]))
        preds0 = classifier1(torch.stack(samples[1][0]))
        labels0 = torch.argmax(preds0, dim=-1).reshape(100,preds0.shape[1])
        labels1 = torch.argmax(preds1, dim=-1).reshape(100,preds1.shape[1])
        classes_mul = torch.stack([classes[0][:100] for _ in np.arange(100)]).cuda()
        print(classes_mul.shape, labels1.shape)
        acc1 = torch.mean((labels1 == classes_mul)*1.)
        acc0 = torch.mean((labels0 == classes_mul)*1.)

        r, range, bins = self.extract_hist_values(samples)
        sm =  {'neg_entropy' : negative_entropy(r.cpu(), range, bins), 'acc0' :acc0, 'acc1': acc1}
        update_details(sm,m)
        return sm

