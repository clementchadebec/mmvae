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
from ..multi_vaes import Multi_VAES

from utils import get_mean, kl_divergence, add_channels, adjust_shape
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples
from torchvision.utils import save_image


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)




class JMVAE_NF(Multi_VAES):
    def __init__(self,params, joint_encoder, vaes):
        super(JMVAE_NF, self).__init__(params, vaes)
        self.joint_encoder = joint_encoder
        self.qz_xy = dist_dict[params.dist]
        self.qz_xy_params = None # populated in forward
        self.beta_kl = params.beta_kl
        self.max_epochs = params.epochs
        self.fix_jencoder = params.fix_jencoder
        self.fix_decoders = params.fix_decoders
        self.lik_scaling = (1,1)
        self.decrease_beta_kl = params.decrease_beta_kl # how much to decrease
        self.ratio_kl_recon = [None,None]
        self.no_recon = params.no_recon # if we want to omit the reconstruction term in the loss (jmvae loss)



    def forward(self, x):
        """ Using the joint encoder, it computes the latent representation and returns
                qz_xy, pxy_z, z"""

        self.qz_xy_params = self.joint_encoder(x)

        qz_xy = self.qz_xy(*self.qz_xy_params)
        z_xy = qz_xy.rsample()
        recons = []
        for m, vae in enumerate(self.vaes):
            recons.append(vae.decoder(z_xy)["reconstruction"])

        return qz_xy, recons, z_xy

    def _compute_kld(self, x):

        """ Computes KL(q(z|x_m) || q(z|x,y)) (stochastic approx in z)"""

        self.qz_xz_params = self.joint_encoder(x)
        qz_xy = self.qz_xy(*self.qz_xz_params)
        kld = 0
        for m,vae in enumerate(self.vaes):
            r,z0,mu,log_var,z,log_abs_det_jac = vae.forward(x[m]).to_tuple()
            kld = kld- qz_xy.log_prob(z).sum(1)

            log_q_z0 = (
                    -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
            ).sum(dim=1)

            kld = kld + log_q_z0 - log_abs_det_jac

        return kld.mean()

    def compute_kld(self, x):
        """ Computes KL(q(z|x,y) || q(z|x)) + KL(q(z|x,y) || q(z|y))
        We also add terms to avoid q(z|x) spreading out too much"""


        qz_xy,_,z_xy = self.forward(x)
        reg = 0
        details_reg = {}
        for m, vae in enumerate(self.vaes):
            flow_output = vae.iaf_flow(z_xy) if hasattr(vae, "iaf_flow") else vae.inverse_flow(z_xy)
            vae_output = vae.forward(x[m])
            mu, log_var, z0 = vae_output.mu, vae_output.log_var, flow_output.out
            log_q_z0 = (-0.5 * (log_var + np.log(2*np.pi) + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)

            # kld -= log_q_z0 + flow_output.log_abs_det_jac
            details_reg[f'recon_loss_{m}'] = vae_output.recon_loss.sum() * x[m].shape[0] # already the negative log conditional expectation
            details_reg[f'kld_{m}'] = qz_xy.log_prob(z_xy).sum() - (log_q_z0 + flow_output.log_abs_det_jac).sum()
            if self.ratio_kl_recon[m] is None:
                if self.no_recon :
                    self.ratio_kl_recon[m] = 0
                else:
                    self.ratio_kl_recon[m] = details_reg[f'kld_{m}'].item() / details_reg[f'recon_loss_{m}'].item()
            reg += (self.beta_kl*details_reg[f'kld_{m}'] + self.ratio_kl_recon[m]*details_reg[f'recon_loss_{m}'])* self.lik_scaling[m]

        return reg, details_reg


    def reconstruct(self, data, runPath,epoch):
        self.eval()
        _data = [d[:8] for d in data]
        with torch.no_grad():
            _ , recons, _ = self.forward(_data)
        for m, recon in enumerate(recons):
            d = data[m][:8].cpu()
            recon = recon.squeeze(0).cpu()
            comp = torch.cat([d,recon])
            filename = '{}/recon_{}_{:03d}.png'.format(runPath, m, epoch)
            save_image(comp, filename)
            wandb.log({f'recon_{m}' : wandb.Image(filename)})

        return


    def analyse_joint_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        qz_xy, _, zxy = self.forward(bdata)
        m,s = qz_xy.mean, qz_xy.stddev
        zxy = zxy.reshape(-1,zxy.size(-1))
        return m,s, zxy.cpu().numpy()


    def step(self, epoch):
        """ Change the hyperparameters of the models"""
        if epoch >= self.params.warmup :
            self.beta_kl *=self.decrease_beta_kl
        wandb.log({'beta_kl' : self.beta_kl})


