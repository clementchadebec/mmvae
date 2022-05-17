# Base MMVAE class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}


class MMVAE(nn.Module):
    def __init__(self, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = dist_dict[params.dist]
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pz_params = None  # defined in subclass
        self.align = -1

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            # encode each modality with its specific encoder
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions : reconstruction of modality 1 given modality 2 / given modality 1 etc...
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def analyse(self, data, K):
        """Compute all embeddings, and plot empirical posterior and prior distribution.
        It also computes KL distances between distributions"""
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params) # prior

            # Add prior samples to the samples from each encoder generated during the forward pass
            # zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
            #        *[zs.permute(1,0,2).reshape(-1, zs.size(-1)) for zs in zss]]

            zss = [*[zs.permute(1,0,2).reshape(-1, zs.size(-1)) for zs in zss]] # No prior samples
            # Labels
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )

        return torch.cat(zss, 0).cpu().numpy(), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df

            # previously embed_umap(torch.cat(zss, 0).cpu().numpy()) but incompatibility with u_map version

    def analyse_posterior(self,data, n_samples=8):
        """ For all points in data, computes the mean and standard of q(z|x), q(z|y)"""
        bdata = [d[:n_samples] for d in data]
        qz_xs, _, _ = self.forward(bdata)
        means, std = [], []
        for m, qz_x in enumerate(qz_xs):
            means.append(get_mean(qz_x))
            std.append(qz_x.stddev)
        return means, std


    def sample_from_conditional(self,data,n = 10):
        self.eval()
        px_zss = [self.forward(data)[1] for _ in range(n)]
        with torch.no_grad():
            # cross-modal matrix of reconstructions : reconstruction of modality 1 given modality 2 / given modality 1 etc...
            recons = [torch.stack([torch.stack([get_mean(px_z) for px_z in r]) for r in px_zs]) for px_zs in px_zss]
            recons = torch.stack(recons).squeeze().permute(1,2,0,3,4,5)

        return recons