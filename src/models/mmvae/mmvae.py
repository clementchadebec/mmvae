# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import wandb
from analysis.pytorch_fid import get_activations,calculate_activation_statistics,calculate_frechet_distance
from analysis.pytorch_fid.inception import InceptionV3
from torchvision import transforms
from dataloaders import MultimodalBasicDataset
from .multi_vaes import Multi_VAES

from utils import get_mean, kl_divergence, add_channels, adjust_shape
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples
from torchvision.utils import save_image


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)




class MMVAE(Multi_VAES):
    def __init__(self,params, vaes):
        super(MMVAE, self).__init__(params, vaes)
        self.qz_x = dist_dict[params.dist] # We use the same distribution for both modalities
        self.px_z = dist_dict[params.dist]
        device = 'cuda' if params.cuda else 'cpu'
        self.px_z_std = torch.tensor(0.75).to(device)
        self.lik_scaling = ((3*32*32)/(1*28*28), 1) if params.llik_scaling ==0 else (params.llik_scaling, 1)

    def forward(self, x, K=1):
        """ Using the unimodal encoders, it computes qz_xs, px_zs, zxs"""

        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        qz_x_params =[]
        for m, vae in enumerate(self.vaes):
            # encode each modality with its specific encoder
            # print(torch.cat([x[m] for _ in range(K)]).shape)
            o =  vae(torch.cat([x[m] for _ in range(K)]))
            # print(o.mu.shape)
            mu, std = o.mu.reshape(K,len(x[m]), -1), torch.exp(0.5*o.log_var).reshape(K,len(x[m]),-1)
            qz_x_params.append((mu,std))
            # print(m,torch.max(o.log_var))
            qz_xs.append(self.qz_x(mu, std))
            zss.append(o.z.reshape(K,len(x[m]),*o.z.shape[1:]))
            px_zs[m][m] = o.recon_x.reshape(K,len(x[m]),*o.recon_x.shape[1:])  # fill-in diagonal
            px_zs[m][m]= self.px_z(px_zs[m][m], self.px_z_std)
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    zs_resh = zs.reshape(zs.shape[0]*zs.shape[1],-1)
                    px_zs[e][d] = vae.decoder(zs_resh).reconstruction
                    px_zs[e][d] = px_zs[e][d].reshape(K,zs.shape[1],*px_zs[e][d].shape[1:])
                    px_zs[e][d] = self.px_z(px_zs[e][d], self.px_z_std )

        return qz_xs, px_zs, zss, qz_x_params



    def reconstruct(self, data, runPath, epoch):
        pass


    def analyse_joint_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        qz_xy, _, zxy, _ = self.forward(bdata)

        zxy = 1/2 * (zxy[0] + zxy[1])
        m = (qz_xy[0].mean + qz_xy[1].mean)/2

        zxy = zxy.reshape(-1,zxy.size(-1))
        return m,None, zxy.cpu().numpy()



