''' MOEpoE implementation '''


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




class MOEPOE(Multi_VAES):

    def __init__(self,params, vaes):

        super(MOEPOE, self).__init__(params, vaes)
        self.lik_scaling = [1,1]
        self.qz_x = dist.Normal


    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        mus = []
        log_vars = []

        for m, vae in enumerate(self.vaes):

            o = vae(x[m])
            mus.append(o.mu)
            log_vars.append(o.log_var)


        # Add the prior and the joint product to the mixture of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars]) # Compute the inverse of variances
        joint_lnV = - torch.logsumexp(lnT, dim=0) # variances of the product of expert

        tensor_mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT)*tensor_mus).sum(dim=0)*torch.exp(joint_lnV)

        mus.append(joint_mu)
        log_vars.append(joint_lnV)

        # Compute the Elbo for each of this sampling distributions
        elbos = 0
        zs = 0
        for i, mu in enumerate(mus):
            q = dist.Normal(mu, torch.exp(0.5*log_vars[i]))
            # Sample from the distribution
            z = q.rsample()
            zs += z
            # Decode in each modality
            for m, vae in enumerate(self.vaes):
                recon = vae.decoder(z)['reconstruction']

                # Decoder distribution is assumed to be a gaussian
                lpx_z = -1/2*torch.sum((recon-x[m])**2)*self.lik_scaling[m]
                elbos += lpx_z

            # And compute the KLD
            kld = kl_divergence(q, self.pz(*self.pz_params))
            elbos -= kld.sum()

        elbos = elbos/len(mus) # divide by the number of product of experts
        zs = zs/len(mus)

        res_dict = dict(
            elbo = elbos/len(x),
            z_joint = zs
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