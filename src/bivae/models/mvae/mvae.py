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

    def __init__(self,params, vaes):

        super(MVAE, self).__init__(params, vaes)
        self.lik_scaling = [1,1]


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
        lnT = torch.stack([-l for l in log_vars]) # Compute the inverse of variances
        lnV = - torch.logsumexp(lnT, dim=0) # variances of the product of expert

        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT)*mus).sum(dim=0)*torch.exp(lnV)

        joint_std = torch.exp(0.5*lnV)

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
            lpx_z = torch.sum(-1/2*(recon - x[m])**2)*self.lik_scaling[m]
            kld = kl_divergence(dist.Normal(mus[m], torch.exp(0.5*log_vars[m])), dist.Normal(*self.pz_params))

            elbo += lpx_z -kld.sum()

            # The joint elbo reconstruction term
            elbo += torch.sum(-1/2*(joint_recons[m] - x[m])**2)*self.lik_scaling[m]

        # Joint KLdivergence
        kld = kl_divergence(dist.Normal(joint_mu, joint_std), dist.Normal(*self.pz_params))
        elbo -= kld.sum()

        res_dict = dict(
            elbo = elbo/len(x),
            z_joint = z_joint
        )

        return res_dict







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