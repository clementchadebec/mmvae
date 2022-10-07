# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.distributions as dist
import wandb

from ..multi_vaes import Multi_VAES
from tqdm import tqdm
from bivae.utils import get_mean, kl_divergence, add_channels, adjust_shape, unpack_data, update_details
from torchvision.utils import save_image
import torch.nn.functional as F


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
        self.no_recon = params.no_recon if hasattr(params, 'no_recon') else False# if we want to omit the reconstruction term in the loss (jmvae loss)
        self.train_latents = None




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
            details_reg[f'recon_loss_{m}'] = self.compute_recon_loss(x[m],vae_output.recon_x,m) # already the negative log conditional expectation
            details_reg[f'kld_{m}'] = qz_xy.log_prob(z_xy).sum() - (log_q_z0 + flow_output.log_abs_det_jac).sum()
            if self.ratio_kl_recon[m] is None:
                if self.no_recon :
                    self.ratio_kl_recon[m] = 0
                else:
                    self.ratio_kl_recon[m] = details_reg[f'kld_{m}'].item() / details_reg[f'recon_loss_{m}'].item()
            reg += (self.beta_kl*details_reg[f'kld_{m}'] + self.ratio_kl_recon[m]*details_reg[f'recon_loss_{m}'])* self.lik_scaling[m]

        return reg, details_reg

    def compute_joint_ll_from_uni(self, data, cond_mod, K=1000, batch_size_K = 100):

        '''
                Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with MonteCarlo Sampling and the approximation :

                ln p(x,y) = \sum_{z ~ q(z|y)} ln p(x|z)p(y|z)p(z)/q(z|y)

                '''


        # Then iter on each datapoint to compute the iwae estimate of ln(p(x|y))
        ll = 0
        for i in range(len(data[0])):
            start_idx, stop_index = 0, batch_size_K
            lnpxs = []
            repeated_data_point = torch.stack(batch_size_K * [data[cond_mod][i]]) # batch_size_K, n_channels, h, w

            while stop_index <= K:

                # Encode with the conditional VAE
                output = self.vaes[cond_mod](repeated_data_point)
                latents = output.z  # (batchsize_K, latent_dim)

                # Decode in each modality
                lpxy_z = 0
                for m, vae in enumerate(self.vaes):
                    recon = self.vaes[m].decoder(latents).reconstruction

                    # Compute lnp(y|z)
                    lpxy_z += -0.5 * torch.sum((recon - data[m][i])**2, dim=(1, 2, 3)) - np.prod(data[m][i].shape) / 2 * np.log(
                        2 * np.pi)

                # Compute lpz
                prior = self.pz(*self.pz_params)
                lpz = torch.sum(prior.log_prob(latents), dim=-1)

                # Finally compute lqz_x
                mu, log_var, z0 = output.mu, output.log_var, output.z0
                log_q_z0 = (-0.5 * (log_var + np.log(2 * np.pi) + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(
                    dim=1)
                lqz_x = log_q_z0 - output.log_abs_det_jac

                lnpxs.append(torch.logsumexp(lpxy_z + lpz - lqz_x,dim=0))
                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0)

        return {f'joint_ll_from_{cond_mod}': ll / len(data[0])}


    def compute_recon_loss(self,x,recon,m):
        return F.mse_loss(x.reshape(x.shape[0],-1),
                          recon.reshape(x.shape[0],-1),reduction='sum')


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


    def compute_all_train_latents(self,train_loader):
        mu = []
        labels = []
        with torch.no_grad():
            for i, dataT in enumerate(tqdm(train_loader)):
                data = unpack_data(dataT, device=self.params.device)
                mu_data = self.joint_encoder(data)[0]
                mu.append(mu_data)
                labels.append(dataT[0][1].to(self.params.device))

        self.train_latents = torch.cat(mu), torch.cat(labels)


    def compute_joint_likelihood(self,data, K=1000, batch_size_K=100):

        # First compute all the parameters of the joint posterior q(z|x,y)
        self.qz_xy_params = self.joint_encoder(data)
        qz_xy = self.qz_xy(*self.qz_xy_params)

        # Then sample K samples for each distribution
        z_xy = qz_xy.rsample([K]) # (K, n_data_points, latent_dim)
        z_xy = z_xy.permute(1,0,2)

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
                qz_xy = self.qz_xy(self.qz_xy_params[0][i], self.qz_xy_params[1][i])
                lqz_xy = torch.sum(qz_xy.log_prob(latents), dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=-1)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0)

        return {'likelihood' : ll/len(data[0])}


    def compute_conditional_likelihood(self, data, cond_mod, gen_mod, K=1000, batch_size_K=100):

        '''
                Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with MonteCarlo Sampling and the approximation :

                ln p(x|y) = \sum_{z ~ q(z|y)} ln p(x|z)

                '''


        # Then iter on each datapoint to compute the iwae estimate of ln(p(x|y))
        ll = 0
        for i in range(len(data[0])):
            start_idx, stop_index = 0, batch_size_K
            lnpxs = []
            repeated_data_point = torch.stack(batch_size_K * [data[cond_mod][i]]) # batch_size_K, n_channels, h, w

            while stop_index <= K:

                # Encode with the conditional VAE
                latents = self.vaes[cond_mod](repeated_data_point).z  # (batchsize_K, latent_dim)

                # Decode with the opposite decoder
                recon = self.vaes[gen_mod].decoder(latents).reconstruction

                # Compute lnp(y|z)
                lpx_z = -0.5 * torch.sum((recon - data[gen_mod][i])**2, dim=(1, 2, 3)) - np.prod(data[gen_mod][i].shape) / 2 * np.log(
                    2 * np.pi)

                lnpxs.append(torch.logsumexp(lpx_z,dim=0))
                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0)

        return {f'cond_likelihood_{cond_mod}_{gen_mod}': ll / len(data[0])}


