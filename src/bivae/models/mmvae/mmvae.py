# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from ..multi_vaes import Multi_VAES
from bivae.utils import Constants, unpack_data, update_details
from numpy.random import randint

from torchvision.utils import save_image
from tqdm import tqdm

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)




class MMVAE(Multi_VAES):
    def __init__(self,params, vaes):
        super(MMVAE, self).__init__(params, vaes)
        self.qz_x = dist_dict[params.dist] # We use the same distribution for both modalities
        self.px_z = dist_dict[params.dist]
        device = params.device
        self.px_z_std = torch.tensor(0.75).to(device)
        self.train_latents = None

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
            mu =  o.mu.reshape(K,len(x[m]), -1)
            std =  o.std.reshape(K,len(x[m]), -1)
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

    def step(self, epoch):
        return

    def compute_all_train_latents(self, train_loader):
        mu = []
        labels = []
        with torch.no_grad():
            for i, dataT in enumerate(tqdm(train_loader)):
                data = unpack_data(dataT, device=self.params.device)
                idx = randint(2) # q(z|x,y) = 1/2(q(z|x) + q(z|y))
                mu_data = self.vaes[idx].encoder(data[idx])[0]
                mu.append(mu_data)
                labels.append(dataT[0][1].to(self.params.device))
        self.train_latents = torch.cat(mu), torch.cat(labels)


    def compute_qz_x_params(self,data):
        outputs_encoders = [self.vaes[m].encoder(data[m]) for m in range(len(data))]
        qz_xy_params = [(o['embedding'], torch.exp(0.5*o['log_covariance'])) for o in outputs_encoders]
        return qz_xy_params



    def compute_joint_likelihood(self,data, K=1000, batch_size_K=100):

        # First compute all the parameters of the joint posterior q(z|x,y)
        self.qz_xy_params = self.compute_qz_x_params(data)
        qz_xs = [self.qz_x(*params) for params in self.qz_xy_params]

        # Then sample K samples for each distribution
        bernouillis = np.random.binomial(1,1/2, size=K*len(data[0]))
        bern_repet = np.stack(self.params.latent_dim * [bernouillis]).T # (K*n_data_points , latent_dim)
        bern = torch.from_numpy(bern_repet.reshape(K,len(data[0]), -1)).cuda()
        # print(bern[0,0])

        z_xy =  bern * qz_xs[0].rsample([K]) + (1-bern) * qz_xs[1].rsample([K]) # (K, n_data_points, latent_dim)
        z_xy = z_xy.permute(1,0,2)
        # print(z_xy.shape)

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(len(data[0])):
            start_idx,stop_index = 0,batch_size_K
            lnpxs = []
            while stop_index <= K:
                latents = z_xy[i][start_idx:stop_index]

                # Compute p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0 # ln(p(x,y|z))
                for m, vae in enumerate(self.vaes):

                    mus = vae.decoder(latents)['reconstruction'] # (batch_size_K, nb_channels, w, h)
                    x_m = data[m][i] # (nb_channels, w, h)
                    lpx_z = -0.5 * torch.sum((mus - x_m)**2,dim=(1,2,3)) - np.prod(x_m.shape)/2*np.log(2*np.pi)
                    lpx_zs += lpx_z

                # Compute ln(p(z))
                prior = self.pz(*self.pz_params)
                lpz = torch.sum(prior.log_prob(latents), dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xs = [self.qz_x(params[0][i], params[1][i]) for params in self.qz_xy_params]
                lqz_xs = torch.stack([torch.sum(q.log_prob(latents), dim=-1) for q in qz_xs])
                # print(lqz_xs.shape)
                lqz_xy = torch.logsumexp(lqz_xs, dim=0)/2
                # print(lqz_xy.shape)
                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=-1)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0)

        return {'likelihood' : ll/len(data[0])}


    def compute_conditional_likelihood(self,data, cond_mod, gen_mod,K=1000,batch_size_K = 100):

        '''
        Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with MonteCarlo Sampling and the approximation :

        ln p(x|y) = \sum_{z ~ q(z|y)} ln p(x|z)

        '''

        o = self.vaes[cond_mod].encoder(data[cond_mod])
        qz_xy_params = (o['embedding'], torch.exp(0.5 * o['log_covariance']))

        qz_xs = self.qz_x(*qz_xy_params)
        # Sample from the conditional encoder distribution
        z_x = qz_xs.rsample([K]).permute(1,0,2) # n_data_points,K,latent_dim

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x|y))
        ll = 0
        for i in range(len(data[0])):
            start_idx, stop_index = 0, batch_size_K
            lnpxs = []
            while stop_index <= K:
                latents = z_x[i][start_idx:stop_index]

                # Compute p(x_m|z) for z in latents and for each modality m
                mus = self.vaes[gen_mod].decoder(latents)['reconstruction']  # (batch_size_K, nb_channels, w, h)
                x_m = data[gen_mod][i]  # (nb_channels, w, h)
                lpx_z = -0.5 * torch.sum((mus - x_m) ** 2, dim=(1, 2, 3)) - np.prod(x_m.shape) / 2 * np.log(
                    2 * np.pi)

                lnpxs.append(torch.logsumexp(lpx_z, dim=0))

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0)

        return {f'cond_likelihood_{cond_mod}_{gen_mod}' : ll/len(data[0])}

    def compute_conditional_likelihoods(self, data, K=1000, batch_size_K=100):

        metrics = self.compute_conditional_likelihood(data, 0,1, K, batch_size_K)
        update_details(metrics, self.compute_conditional_likelihood(data, 1, 0,K, batch_size_K))
        return metrics


