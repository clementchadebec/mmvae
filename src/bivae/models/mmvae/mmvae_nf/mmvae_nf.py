''' Adding normalizing flows to the MMVAE '''

# Base JMVAE-NF class definition

import numpy as np
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



    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        outputs = [None, None]
        recons = [[None,None], [None,None]]
        zs = [None, None]
        ln_qz_xs = [[None, None], [None,None]]
        for m, vae in enumerate(self.vaes):

            o = vae(x[m])
            outputs[m] = o
            recons[m][m] = o.recon_x
            zs[m] = o.z
            log_prob_z0 = (
                    -0.5 * (o.log_var + torch.pow(o.z0 - o.mu, 2) / torch.exp(o.log_var))
            ).sum(dim=1)
            ln_qz_xs[m][m] = log_prob_z0 - o.log_abs_det_jac


        # Fill outside the diagonals
        for e,z in enumerate(zs):
            for d, vae in enumerate(self.vaes):
                if e!=d:

                    # First compute the cross reconstruction
                    recons[e][d] = vae.decoder(z).reconstruction

                    # Then compute ln q(z|y)
                    flow_out = vae.iaf_flow(z)
                    z0 = flow_out.out
                    mu, log_var = outputs[d].mu, outputs[d].log_var
                    log_prob_z0 = (
                    -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)
                    ln_qz_xs[e][d] = log_prob_z0 + flow_out.log_abs_det_jac

        return ln_qz_xs, zs, recons




