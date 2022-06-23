# Base Joint Multimodal VAEs. It inherits MMVAE but it has an additional joint auto_encoder

from itertools import combinations

import torch
import torch.distributions as dist

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df
from .mmvae import MMVAE

dist_dict = {'normal' : dist.Normal, 'laplace' : dist.Normal}

class JMVAE():
    def __init__(self, params, joint_encoder, vaes):
        self.joint_encoder = joint_encoder
        self.qz_xy = dist_dict[params.dist]
        self.qz_xz_params = None # populated in forward
        self.vaes = vaes # a list of vaes already defined

    def eval(self):
        for vae in self.vaes:
            vae.eval()
        self.joint_encoder.eval()

    # One additional method to forward using the joint autoencoder
    def forward_joint(self, x, K=1):
        """ Using the joint encoder, it computes the latent representation and returns
        qz_xy, pxy_z, z"""

        self.qz_xz_params = self.joint_encoder(x)
        qz_xy = self.qz_xy(*self.qz_xz_params)
        z_xy = qz_xy.rsample(torch.Size([K]))
        pxy_z = []
        for m, vae in enumerate(self.vaes):
            px_z = vae.px_z(*vae.dec(z_xy))
            pxy_z.append(px_z)

        return qz_xy, pxy_z,z_xy

    def analyse_joint_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        qz_xy, _, zxy = self.forward_joint(bdata)
        m,s = qz_xy.mean, qz_xy.stddev
        zxy = zxy.permute(1,0,2).reshape(-1,zxy.size(-1))
        return m,s, zxy.cpu().numpy()

    def reconstruct_jointly(self, data):
        self.eval()
        with torch.no_grad():
            _,pxy_z,_ = self.forward_joint(data)
            recons = [get_mean(px_z) for px_z in pxy_z]
        return recons




