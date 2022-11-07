''' MVAE implementation '''

import numpy as np
from numpy.random import randint
import torch
import torch.distributions as dist
import torch.nn.functional as F
import wandb
from torchvision.utils import save_image
from tqdm import tqdm

from bivae.models.multi_vaes import Multi_VAES
from bivae.utils import unpack_data
from bivae.objectives import kl_divergence


class MVAE(Multi_VAES):

    def __init__(self, params, vaes):

        super(MVAE, self).__init__(params, vaes)
        self.lik_scaling = [1, 1]

    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        mus = []
        log_vars = []
        uni_recons = []
        zs = []
        for m, vae in enumerate(self.vaes):
            o = vae(x[m])
            mus.append(o.mu)
            log_vars.append(o.log_var)
            uni_recons.append(o.recon_x)
            zs.append(o.z)

        # Add the prior to the product of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        lnV = - torch.logsumexp(lnT, dim=0)  # variances of the product of expert

        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        joint_std = torch.exp(0.5 * lnV)

        # Sample from the joint posterior
        z_joint = dist.Normal(joint_mu, joint_std).rsample()

        # Decode in each modality
        joint_recons = []
        for m, vae in enumerate(self.vaes):
            joint_recons.append(vae.decoder(z_joint)['reconstruction'])

        # Compute the ELBOS

        elbo = 0

        # The unimodal elbos
        for m, recon in enumerate(uni_recons):
            lpx_z = torch.sum(-1 / 2 * (recon - x[m]) ** 2) * self.lik_scaling[m]
            kld = kl_divergence(dist.Normal(mus[m], torch.exp(0.5 * log_vars[m])), dist.Normal(*self.pz_params))

            elbo += lpx_z - kld.sum()

            # The joint elbo reconstruction term
            elbo += torch.sum(-1 / 2 * (joint_recons[m] - x[m]) ** 2) * self.lik_scaling[m]

        # Joint KLdivergence
        kld = kl_divergence(dist.Normal(joint_mu, joint_std), dist.Normal(*self.pz_params))
        elbo -= kld.sum()

        res_dict = dict(
            elbo=elbo / len(x),
            z_joint=z_joint,
            joint_mu=joint_mu,
            joint_std=joint_std
        )

        return res_dict

    def step(self, epoch):
        return

    def compute_all_train_latents(self, train_loader):
        mu = []
        labels = []
        with torch.no_grad():
            print("Computing all latents variables for the train set")
            for i, dataT in enumerate(tqdm(train_loader)):
                data = unpack_data(dataT, device=self.params.device)
                z = self.forward(data)['z_joint']
                mu.append(z)
                labels.append(dataT[0][1].to(self.params.device))
        self.train_latents = torch.cat(mu), torch.cat(labels)

    def reconstruct(self, data, runPath, epoch):
        """ Reconstruction is not defined for the mvae model since
        the conditional contains all the information"""
        pass

    def compute_joint_ll_from_uni(self, data, cond_mod, K=1000, batch_size_K=100):

        '''
        Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with the same approximation as in MVAE, JMVAE :

        ln p(x|y) = ln E_{q(z|y)}( p(x,y,z)/q(z|y) ) - ln E_{p(z)}(p(y|z))

        Each term is computed using importance sampling. In this function we only compute
        the first term
        '''

        o = self.vaes[cond_mod](data[cond_mod])
        qz_xy_params = (o.mu, o.std)

        qz_xs = self.qz_x(*qz_xy_params)
        # Sample from the conditional encoder distribution
        z_x = qz_xs.rsample([K]).permute(1, 0, 2)  # n_data_points,K,latent_dim

        ll = 0
        for i in range(len(data[0])):
            start_idx, stop_index = 0, batch_size_K
            ln_pxs = []
            while stop_index <= K:
                latents = z_x[i][start_idx:stop_index]

                lpxs_z = 0
                # Compute p(x|z) for z in latents and for each modality m
                for m, vae in enumerate(self.vaes):
                    mus = self.vaes[m].decoder(latents)['reconstruction']  # (batch_size_K, nb_channels, w, h)
                    x_m = data[m][i]  # (nb_channels, w, h)

                    # Compute lnp(y|z)
                    if self.px_z[m] == dist.Bernoulli:
                        lp = self.px_z[m](mus).log_prob(x_m).sum(dim=(1, 2, 3))
                    else:
                        lp = self.px_z[m](mus, scale=1).log_prob(x_m).sum(dim=(1, 2, 3))

                    lpxs_z += lp

                # Prior distribution p(z)
                lpz = torch.sum(self.pz(*self.pz_params).log_prob(latents), dim=-1)

                # Compute the log prob of the posterior q(z|cond_mod)
                lqz_cond_mod = self.qz_x(qz_xy_params[0][i], qz_xy_params[1][i]).log_prob(latents)
                lqz_cond_mod = torch.sum(lqz_cond_mod, dim=-1)

                ln_pxs.append(torch.logsumexp(lpxs_z + + lpz - lqz_cond_mod, dim=0))

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(ln_pxs), dim=0) - torch.log(K)

        return {f'joint_ll_from_{cond_mod}': ll / len(data[0])}

    def compute_joint_likelihood(self, data, K=1000, batch_size_K=100):

        """ Computes the joint likelihood based in the importance sampling formula
        E_q(z|x,y) [ ln  \sum_i p_\theta (x,y,z_i)/q_(z_i|x,y)]"""

        # First we need to compute the joint posterior parameter for each data point

        output = self(data)
        mus = output['joint_mu']
        stds = output['joint_std']

        ll = 0
        # Then iter on each data_point
        for i in range(len(data[0])):

            lnpxs = []
            for _ in range(K // batch_size_K):
                q = dist.Normal(mus[i], stds[i])
                # Sample latent variables
                latents = q.rsample([batch_size_K])

                # Compute reconstruction errors
                lpx_zs = 0
                for m, vae in enumerate(self.vaes):
                    mus = vae.decoder(latents)['reconstruction']  # (batch_size_K, nb_channels, w, h)
                    x_m = data[m][i]  # (nb_channels, w, h)

                    # Compute lnp(y|z)
                    if self.px_z[m] == dist.Bernoulli:
                        lpx_zs += self.px_z[m](mus).log_prob(x_m).sum(dim=(1, 2, 3))
                    else:
                        lpx_zs += self.px_z[m](mus, scale=1).log_prob(x_m).sum(dim=(1, 2, 3))

                # Compute ln(p(z))
                prior = self.pz(*self.pz_params)
                lpz = torch.sum(prior.log_prob(latents), dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                lqz_xs = torch.sum(q.log_prob(latents), dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xs, dim=-1)
                lnpxs.append(ln_px)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - torch.log(K)

        return {'likelihood': ll / len(data[0])}

    