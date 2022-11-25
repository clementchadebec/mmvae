# Base JMVAE-NF class definition


import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import wandb
from bivae.utils import (unpack_data)

from ..multi_vaes import Multi_VAES

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}



class JMVAE_NF(Multi_VAES):
    def __init__(self,params, joint_encoder, vaes):
        super(JMVAE_NF, self).__init__(params, vaes)
        self.joint_encoder = joint_encoder
        self.qz_xy = dist_dict[params.dist]
        self.qz_xy_params = None # populated in forward
        self.beta_kl = params.beta_kl
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



    def compute_kld(self, x):
        """ Computes KL(q(z|x,y) || q(z|x)) + KL(q(z|x,y) || q(z|y))
        We also add terms to avoid q(z|x) spreading out too much"""


        qz_xy,_,z_xy = self.forward(x)
        reg = 0
        details_reg = {}
        for m, vae in enumerate(self.vaes):
            flow_output = vae.flow(z_xy) if hasattr(vae, "flow") else vae.inverse_flow(z_xy)
            vae_output = vae.encoder(x[m])
            mu, log_var, z0 = vae_output.embedding, vae_output.log_covariance, flow_output.out
            log_q_z0 = (-0.5 * (log_var + np.log(2*np.pi) + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)

            # kld -= log_q_z0 + flow_output.log_abs_det_jac
            details_reg[f'kld_{m}'] = qz_xy.log_prob(z_xy).sum() - (log_q_z0 + flow_output.log_abs_det_jac).sum()
            if self.no_recon :
                reg += self.beta_kl*details_reg[f'kld_{m}']
            else:
                vae_output = vae(x[m])
                details_reg[f'recon_loss_{m}'] = self.compute_recon_loss(x[m],vae_output.recon_x,m) # already the negative log conditional expectation
                self.ratio_kl_recon[m] = details_reg[f'kld_{m}'].item() / details_reg[f'recon_loss_{m}'].item()
                reg += ( self.beta_kl*details_reg[f'kld_{m}'] + self.ratio_kl_recon[m]*details_reg[f'recon_loss_{m}']) # I don't think any likelihood scaling is needed here

        return reg, details_reg

    def compute_joint_ll_from_uni(self, data, cond_mod, K=1000, batch_size_K = 100):

        '''
                Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with MonteCarlo Sampling and the approximation :

                ln p(x,y) = ln \sum_{z ~ q(z|y)}  p(x|z)p(y|z)p(z)/q(z|y)

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
                    if self.px_z[m] == dist.Bernoulli :
                        # print(f'compute Bernouilli likelihood for modality {m}')
                        lp =  self.px_z[m](recon).log_prob(data[m][i]).sum(dim=(1, 2, 3))
                    else :
                        # print(f'compute normal for mod {m}')
                        lp =  self.px_z[m](recon, scale = 1).log_prob(data[m][i]).sum(dim=(1, 2, 3))

                    lpxy_z += lp


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

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return {f'joint_ll_from_{cond_mod}': ll / len(data[0])}


        
    def compute_recon_loss(self,x,recon,m):
        """Change the way we compute the reocnstruction, through the filter of DCCA"""
        if hasattr(self,'dcca') and self.params.dcca :
            with torch.no_grad():
                t = self.dcca[m](x).embedding
                recon_t = self.dcca[m](recon).embedding
            return F.mse_loss(t,recon_t,reduction='sum')
        else : 
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
        """Computes the mean joint log likelihood.

        Args:
            data (list): the multimodal data on which to compute the likelihood
            K (int, optional):. Defaults to 1000.
            batch_size_K (int, optional): . Defaults to 100.

        Returns:
            dict : contains likelihood metrics
        """

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

                    # Compute lnp(y|z)
                    if self.px_z[m] == dist.Bernoulli :
                        lp = self.px_z[m](mus).log_prob(x_m).sum(dim=(1, 2, 3))
                    else :
                        lp = self.px_z[m](mus, scale=1).log_prob(x_m).sum(dim=(1, 2, 3))


                    lpx_zs += lp

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

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return {'likelihood' : ll/len(data[0])}
    
    def sample_from_moe_subset(self, subset : list ,data : list):
        """Sample z from the mixture of posteriors from the subset.
        Torch no grad is activated, so that no gradient are computed durin the forward pass of the encoders.

        Args:
            subset (list): the modalities to condition on
            data (list): The data 
            K (int) : the number of samples per datapoint
        """
        # Choose randomly one modality for each sample
        
        indices = np.random.choice(subset, size=len(data[0])) 
        zs = torch.zeros((len(data[0]), self.params.latent_dim)).to(self.params.device)
        
        for m in subset:
            with torch.no_grad():
                z = self.vaes[m](data[m][indices == m]).z
                zs[indices==m] = z
        return zs
            

    
    def compute_poe_posterior(self, subset : list,z_ : torch.Tensor,data : list, divide_prior = False):
        """Compute the log density of the product of experts for Hamiltonian sampling.

        Args:
            subset (list): the modalities of the poe posterior
            z_ (torch.Tensor): the latent variables (len(data[0]), latent_dim)
            data (list): _description_
            divide_prior (bool) : wether or not to divide by the prior

        Returns:
            tuple : likelihood and gradients
        """
        
        lnqzs = 0

        z = z_.clone().detach().requires_grad_(True)
        
        if divide_prior:
            # print('Dividing by the prior')
            lnqzs += (0.5* (torch.pow(z,2) + np.log(2*np.pi))).sum(dim=1)
        
        for m in subset:
            # Compute lnqz
            flow_output = self.vaes[m].flow(z) if hasattr(self.vaes[m], "flow") else self.vae[m].inverse_flow(z)
            vae_output = self.vaes[m].encoder(data[m])
            mu, log_var, z0 = vae_output.embedding, vae_output.log_covariance, flow_output.out

            log_q_z0 = (-0.5 * (log_var + np.log(2*np.pi) + torch.pow(z0 - mu, 2) / torch.exp(log_var))).sum(dim=1)
            lnqzs += (log_q_z0 + flow_output.log_abs_det_jac) # n_data_points x 1

        

        g = torch.autograd.grad(lnqzs.sum(), z)[0]
        

            
        return lnqzs, g


    def sample_from_poe_subset(self,subset,data, ax=None, mcmc_steps=100, n_lf=10, eps_lf=0.01, K=1, divide_prior=False):
        """Sample from the product of experts using Hamiltonian sampling.

        Args:
            subset (List[int]): 
            gen_mod (int): 
            data (List[Tensor]): 
            K (int, optional): . Defaults to 100.
        """
        print('starting to sample from poe_subset, divide prior = ', divide_prior)
        
        # Multiply the data to have multiple samples per datapoints
        n_data = len(data[0])
        data = [torch.cat([d]*K) for d in data]
        
        n_samples = len(data[0])
        acc_nbr = torch.zeros(n_samples, 1).to(self.params.device)

        # First we need to sample an initial point from the mixture of experts
        z0 = self.sample_from_moe_subset(subset,data)
        z = z0
        
        # fig, ax = plt.subplots()
        pos = []
        grad = []
        for i in tqdm(range(mcmc_steps)):
            pos.append(z[0].detach().cpu())

            #print(i)
            gamma = torch.randn_like(z, device=self.params.device)
            rho = gamma# / self.beta_zero_sqrt
            
            # Compute ln q(z|X_s)
            ln_q_zxs, g = self.compute_poe_posterior(subset,z,data, divide_prior=divide_prior)
            
            grad.append(g[0].detach().cpu())

            H0 = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2
            # print(H0)
            # print(model.G_inv(z).det())
            for k in range(n_lf):

                #z = z.clone().detach().requires_grad_(True)
                #log_det = G(z).det().log()

                #g = torch.zeros(n_samples, model.latent_dim).cuda()
                #for i in range(n_samples):
                #    g[0] = -grad(log_det, z)[0][0]


                # step 1
                rho_ = rho - (eps_lf / 2) *(-g)

                # step 2
                z = z + eps_lf * rho_

                #z_ = z_.clone().detach().requires_grad_(True)
                #log_det = 0.5 * G(z).det().log()
                #log_det = G(z_).det().log()

                #g = torch.zeros(n_samples, model.latent_dim).cuda()
                #for i in range(n_samples):
                #    g[0] = -grad(log_det, z_)[0][0]

                # Compute the updated gradient
                ln_q_zxs, g = self.compute_poe_posterior(subset,z,data, divide_prior)
                
                #print(g)
                # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                # step 3
                rho__ = rho_ - (eps_lf / 2) * (-g)

                # tempering
                beta_sqrt = 1

                rho =  rho__
                #beta_sqrt_old = beta_sqrt

            H = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2
            # print(H, H0)
    
            alpha = torch.exp(H0-H) 
            # print(alpha)
            

            #print(-log_pi(best_model, z, best_model.G), 0.5 * torch.norm(rho, dim=1) ** 2)
            acc = torch.rand(n_samples).to(self.params.device)
            moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

            acc_nbr += moves

            z = z * moves + (1 - moves) * z0
            z0 = z
            
        pos = torch.stack(pos)
        grad = torch.stack(grad)
        if ax is not None:
            ax.plot(pos[:,0], pos[:,1])
            ax.quiver(pos[:,0], pos[:,1], grad[:,0], grad[:,1])

            # plt.savefig('monitor_hmc.png')
        # 1/0
        print(acc_nbr[:10]/mcmc_steps)
        z = z.detach().resize(K,n_data,self.params.latent_dim)
        return z.detach()
        
        
    
    
