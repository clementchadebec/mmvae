''' Adding normalizing flows to the MMVAE '''

# Base JMVAE-NF class definition

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




class MMVAE_NF(Multi_VAES):

    def __init__(self,params, vaes):

        super(MMVAE_NF, self).__init__(params, vaes)

        assert params.dist == 'normal' # This model assumes gaussian prior and posteriors


    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        outputs = [None, None]
        recons = [[None,None], [None,None]]
        zs = [None, None]
        ln_qz_xs = [None, None]
        for m, vae in enumerate(self.vaes):

            o = vae(x[m])
            outputs[m] = o
            recons[m][m] = o.recon_x
            zs[m] = o.z
            log_prob_z0 = (
                    -0.5 * (o.log_var + torch.pow(o.z0 - o.mu, 2) / torch.exp(o.log_var))
            ).sum(dim=1)
            ln_qz_xs[m] = log_prob_z0 - o.log_abs_det_jac


        # Fill outside the diagonals
        for e,z in enumerate(zs):
            for d, vae in enumerate(self.vaes):
                if e!=d:

                    # Compute the cross reconstruction
                    recons[e][d] = vae.decoder(z).reconstruction


        return ln_qz_xs, zs, recons

    def step(self, epoch):
        return

    def compute_all_train_latents(self, train_loader):
        mu = []
        labels = []
        with torch.no_grad():
            print("Computing all latents variables for the train set")
            for i, dataT in enumerate(tqdm(train_loader)):
                data = unpack_data(dataT, device=self.params.device)
                idx = randint(2) # q(z|x,y) = 1/2(q(z|x) + q(z|y))
                z = self.vaes[idx](data[idx]).z
                mu.append(z)
                labels.append(dataT[0][1].to(self.params.device))
        self.train_latents = torch.cat(mu), torch.cat(labels)


    def reconstruct(self, data, runPath, epoch):
        """ Reconstruction is not defined for the mmvae model since
        the conditional contains all the information"""
        pass
    
    
    def compute_joint_ll_from_uni(self, data, cond_mod, K=1000, batch_size_K=100):
        pass
    
    def compute_joint_likelihood(self, data, cond_mod, K=1000, batch_size_K=100):
        pass