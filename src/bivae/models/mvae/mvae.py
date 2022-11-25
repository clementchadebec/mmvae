''' MVAE implementation '''

import numpy as np
import torch
import torch.distributions as dist
from tqdm import tqdm

from bivae.models.multi_vaes import Multi_VAES
from bivae.objectives import kl_divergence
from bivae.utils import unpack_data


class MVAE(Multi_VAES):

    def __init__(self, params, vaes):
        assert params.dist == 'normal' # This model assume gaussian prior and posterior


        super(MVAE, self).__init__(params, vaes)
        self.qz_x = dist.Normal
        self.subsampling = False
        self.subsets = []
        
        
    def poe(self, mus_list, log_vars_list, eps = 1e-8):
        
        mus = mus_list.copy()
        log_vars=log_vars_list.copy()
        
        # Add the prior to the product of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        # print(lnT)
        lnV = - torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        # print('lnV',lnV)
        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        joint_std = torch.exp(0.5 * lnV)
        return joint_mu, joint_std
    
    def poe_(self, mus_list, log_vars_list, eps = 1e-8):
        print('using original poe')
        mus_stack = torch.stack(mus_list + [torch.zeros_like(mus_list[0])])
        log_vars_stack = torch.stack(log_vars_list + [torch.zeros_like(mus_list[0])])
        
        var       = torch.exp(log_vars_stack) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mus_stack * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, torch.exp(0.5*pd_logvar)
    
    def kl(self,mu, std):
        return kl_divergence(dist.Normal(mu, std), dist.Normal(*self.pz_params)).sum()
    
    
    def infer_latent_from_mod(self, cond_mod, x):
        o = self.vaes[cond_mod].encoder(x)
        mu, log_var = o.embedding, o.log_covariance
        # poe with prior
        mu, std = self.poe([mu],[log_var]) 
        z = dist.Normal(mu, std).rsample()
        return z
              

    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        elbo = 0
        
        mus_tilde = []
        lnV_tilde = []
        


        for m, vae in enumerate(self.vaes):
            o = vae.encoder(x[m])
            u_mu, u_log_var = o.embedding, o.log_covariance
            # Save the unimodal embedding
            mus_tilde.append(u_mu)
            lnV_tilde.append(u_log_var)
            
            # Compute the unimodal elbos
            mu, std =  self.poe([u_mu], [u_log_var])
            # print(m, mu, std)
            # mu, std = u_mu, torch.exp(0.5*u_log_var)
            z = dist.Normal(mu, std).rsample()
            recon = vae.decoder(z).reconstruction
            elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m] - self.kl(mu, std)

        # Add the joint elbo
        joint_mu, joint_std = self.poe(mus_tilde, lnV_tilde)
        z_joint = dist.Normal(joint_mu, joint_std).rsample()

        # Reconstruction term in each modality
        for m, vae in enumerate(self.vaes):
            recon = (vae.decoder(z_joint)['reconstruction'])
            elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m]
        
        # Joint KL divergence
        elbo -= self.kl(joint_mu, joint_std)
            
        # If using the subsampling paradigm, sample subsets and compute the poe
        if self.subsampling :
            # randomly select k subsets
            subsets = self.subsets[np.random.choice(len(self.subsets), self.k_subsample,replace=False)]
            # print(subsets)

            for s in subsets:
                sub_mus, sub_log_vars = [mus_tilde[i] for i in s], [lnV_tilde[i] for i in s]
                mu, std = self.poe(sub_mus, sub_log_vars)
                sub_z = dist.Normal(mu, std).rsample()
                elbo -= self.kl(mu, std)
                # Reconstruction terms
                for m in s:
                    recon = self.vaes[m].decoder(sub_z).reconstruction
                    elbo += torch.sum(-1 / 2 * (recon - x[m]) ** 2) * self.lik_scaling[m]
                    
            # print('computed subsampled elbos')

        res_dict = dict(
            elbo=elbo,
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
        
        # WARNING : this function is not up to date with the rest of the model. The unimodal posterior 
        # are taken to be the encoders whereas it should be the poe between the encoder and prior

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

            ll += torch.logsumexp(torch.Tensor(ln_pxs), dim=0) - np.log(K)

        return {f'joint_ll_from_{cond_mod}': ll / len(data[0])}

    def compute_joint_likelihood(self, data, K=1000, batch_size_K=100):

        """ Computes the joint likelihood based in the importance sampling formula
        E_q(z|x,y) [ ln  \sum_i p_\theta (x,y,z_i)/q_(z_i|x,y)]"""

        # First we need to compute the joint posterior parameter for each data point

        output = self(data)
        joint_mus = output['joint_mu']
        joint_stds = output['joint_std']

        ll = 0
        # Then iter on each data_point
        for i in range(len(data[0])):

            lnpxs = []
            for _ in range(K // batch_size_K):
                q = dist.Normal(joint_mus[i], joint_stds[i])
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

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return {'likelihood': ll / len(data[0])}

    
    
    def sample_from_poe_subset(self, subset,data,K=1, divide_prior=True):
        """ 
        
        Sample from the conditional using the product of experts.
        

        Args:
            subset (List[int]): the modality to condition on
            data (List[Tensor]): the data to use for conditioning
        """
        
        # First we need to compute the mus log vars for each of the encoding modalities
        
        if not divide_prior:
            print('Override : for mvae model, we divide by the prior.')
            divide_prior=True
        
        mus = []
        log_vars = []
        for m in subset:
            with torch.no_grad():
                o = self.vaes[m].encoder(data[m])
                mus.append(o.embedding)
                log_vars.append(o.log_covariance)
            
        # Compute the PoE
        joint_mu, joint_log_var = self.poe(mus, log_vars)

        # Sample from this distribution
        qz_subset = dist.Normal(joint_mu, torch.exp(0.5*joint_log_var))
        zs = qz_subset.sample([K]) # K x n_data x latent_dim
        
        # Decode in the target modality
        return zs
    
    
