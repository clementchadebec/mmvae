# Base JMVAE-NF class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist

from utils import get_mean, kl_divergence
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader

from .vae_circles import CIRCLES
from .j_circles_discs import Enc as joint_encoder

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)

vae = my_VAE_LinNF
vae_config = VAE_LinNF_Config


class JMVAE_NF(nn.Module):
    def __init__(self, params):
        super(JMVAE_NF, self).__init__()
        self.joint_encoder = joint_encoder(params.latent_dim, params.num_hidden_layers)
        self.qz_xy = dist_dict[params.dist]
        self.qz_xy_params = None # populated in forward
        self.pz = dist_dict[params.dist]
        self.mod = 2
        my_vae_config = vae_config(input_dim=input_dim, latent_dim = params.latent_dim,flows = ['Radial', 'Radial'])
        # my_vae_config = vae_config(input_dim=input_dim, latent_dim = params.latent_dim)

        self.vaes = nn.ModuleList([ vae(model_config=my_vae_config) for _ in range(self.mod)])
        self.modelName = 'jmvae_nf_circles_squares'
        self.params = params
        self.data_path = params.data_path
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, params.latent_dim), requires_grad=False)  # logvar
        ])


    @property
    def pz_params(self):
        return self._pz_params

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        # load base datasets
        t1, s1 = CIRCLES.getDataLoaders(batch_size, 'squares', shuffle, device, data_path=self.data_path)
        t2, s2 = CIRCLES.getDataLoaders(batch_size, 'circles', shuffle, device, data_path=self.data_path)

        train_circles_discs = TensorDataset([t1.dataset, t2.dataset])
        test_circles_discs = TensorDataset([s1.dataset, s2.dataset])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train, test

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
        """ Computes KL(q(z|x,y) || q(z|x))"""
        qz_xy,_,z_xy = self.forward(x)
        kld = 2*qz_xy.log_prob(z_xy).sum(-1)
        for m, vae in enumerate(self.vaes):
            flow_output = vae.iaf_flow(z_xy) if hasattr(vae, "iaf_flow") else vae.inverse_flow(z_xy)
            vae_output = vae.forward(x[m])
            mu, log_var, z0 = vae_output.mu, vae_output.log_var, flow_output.out
            log_q_z0 = (
                    -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
            ).sum(dim=1)
            kld -= log_q_z0 - flow_output.log_abs_det_jac
            # kld += 1/3*vae_output.recon_loss -log_q_z0-flow_output.log_abs_det_jac

        return kld.mean()


    def generate(self,runPath, epoch, N= 10):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                data.append(vae.decoder(latents)["reconstruction"])
        return data  # list of generations---one for each modality

    def reconstruct(self, data, runPath,epoch):
        self.eval()
        _data = [d[:8] for d in data]
        with torch.no_grad():
            _ , recons, _ = self.forward(_data)
        for m, recon in enumerate(recons):
            d = data[m][:8].cpu()
            recon = recon.squeeze(0).cpu()
            comp = torch.cat([d,recon])
            save_image(comp, '{}/recon_{}_{:03d}.png'.format(runPath, m, epoch))
        return

        return

    def analyse_joint_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        qz_xy, _, zxy = self.forward(bdata)
        m,s = qz_xy.mean, qz_xy.stddev
        zxy = zxy.reshape(-1,zxy.size(-1))
        return m,s, zxy.cpu().numpy()

    def analyse_rayons(self,data, r0, r1, runPath, epoch):
        m,s,zxy = self.analyse_joint_posterior(data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_{:03}.png'.format(runPath,epoch))

    def analyse(self, data, runPath, epoch, ticks=None, classes=None):
        m, s, zxy = self.analyse_joint_posterior( data, n_samples=len(data[0]))
        return


    def analyse_posterior(self,data, n_samples, runPath, epoch, ticks=None, N= 30):
        """ For all points in data, samples N points from q(z|x) and q(z|y)"""
        bdata = [d[:n_samples] for d in data]
        #zsamples[m] is of size N, n_samples, latent_dim
        zsamples = [torch.stack([self.vaes[m].forward(bdata[m]).__getitem__('z') for _ in range(N)]) for m in range(self.mod)]
        plot_samples_posteriors(zsamples, '{}/samplepost_{:03d}.png'.format(runPath, epoch), None)
        return


    def sample_from_conditional(self,data, runPath, epoch, n=10):
        bdata = [d[:8] for d in data]
        self.eval()
        samples = [[[],[]],[[],[]]]
        with torch.no_grad():

            for i in range(n):
                o0 = self.vaes[0].forward(bdata[0])
                o1 = self.vaes[1].forward(bdata[1])
                z0 = o0.__getitem__('z')
                z1 = o1.__getitem__('z')
                samples[0][1].append(self.vaes[1].decoder(z0)["reconstruction"])
                samples[1][0].append(self.vaes[0].decoder(z1)["reconstruction"])
                samples[0][0].append(o0.__getitem__('recon_x'))
                samples[1][1].append(o1.__getitem__('recon_x'))

        for r, recon_list in enumerate(samples):
            for o, recon in enumerate(recon_list):

                _data = bdata[r].cpu()
                recon = torch.stack(recon).resize(n*8,1, 32, 32).cpu()
                comp = torch.cat([_data,recon])
                save_image(comp,'{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))