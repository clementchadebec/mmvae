# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
import wandb
from analysis.pytorch_fid import get_activations,calculate_activation_statistics,calculate_frechet_distance
from analysis.pytorch_fid.inception import InceptionV3
from torchvision import transforms
from ..dataloaders import MultimodalBasicDataset


from utils import get_mean, kl_divergence, add_channels
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)




class JMVAE_NF(nn.Module):
    def __init__(self,params, joint_encoder, vaes):
        super(JMVAE_NF, self).__init__()
        self.joint_encoder = joint_encoder
        self.qz_xy = dist_dict[params.dist]
        self.qz_xy_params = None # populated in forward
        self.pz = dist_dict[params.dist]
        self.mod = 2
        self.vaes = vaes
        # self.vaes = nn.ModuleList([ vae(model_config=vae_config) for _ in range(self.mod)])
        self.modelName = None
        self.params = params
        self.data_path = params.data_path
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, params.latent_dim), requires_grad=False)  # logvar
        ])
        self.beta_rec = params.beta_rec
        self.max_epochs = params.epochs
        self.fix_jencoder = params.fix_jencoder
        self.fix_decoders = params.fix_decoders


    @property
    def pz_params(self):
        return self._pz_params

    def getDataLoaders(self):
        raise "getDataLoaders class must be defined in the subclasses"
        return

    def forward(self, x):
        """ Using the joint encoder, it computes the latent representation and returns
                qz_xy, pxy_z, z"""

        # print(list(self.vaes[0].decoder.parameters()))

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
            details_reg[f'recon_loss_{m}'] = vae_output.loss.sum()
            details_reg[f'kld_{m}'] = qz_xy.log_prob(z_xy).sum() - (log_q_z0 + flow_output.log_abs_det_jac).sum()
            reg += details_reg[f'kld_{m}'] + self.beta_rec * details_reg[f'recon_loss_{m}']

        return reg, details_reg


    def generate(self, N= 10):
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

    def analyse(self, data, runPath, epoch, classes=None):
        pass

    def analyse_uni_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        zsamples = [self.vaes[m].forward(bdata[m]).z.cpu() for m in range(self.mod)]
        return zsamples


    def analyse_posterior(self,data, n_samples, runPath, epoch, ticks=None, N= 30):
        """ For all points in data, samples N points from q(z|x) and q(z|y)"""
        bdata = [d[:n_samples] for d in data]
        #zsamples[m] is of size N, n_samples, latent_dim
        zsamples = [torch.stack([self.vaes[m].forward(bdata[m]).__getitem__('z') for _ in range(N)]) for m in range(self.mod)]
        plot_samples_posteriors(zsamples, '{}/samplepost_{:03d}.png'.format(runPath, epoch), None)
        wandb.log({'sample_posteriors' : wandb.Image('{}/samplepost_{:03d}.png'.format(runPath, epoch))})
        return


    def _sample_from_conditional(self,bdata, n=10):
        """Samples from q(z|x) and reconstruct y and conversely"""
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
        return samples

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        bdata = [d[:8] for d in data]
        self.eval()
        samples = self._sample_from_conditional(bdata, n)

        for r, recon_list in enumerate(samples):
            for o, recon in enumerate(recon_list):
                _data = bdata[r].cpu()
                recon = torch.stack(recon)
                _,_,ch,w,h = recon.shape
                recon = recon.resize(n * 8, ch, w, h).cpu()
                comp = torch.cat([_data, recon])
                filename = '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch)
                save_image(comp, filename)
                wandb.log({'cond_samples_{}x{}.png'.format(r,o) : wandb.Image(filename)})

    def compute_fid(self, batchsize, device, dims=2048, nb_batches=20, to_tensor=False):
        if to_tensor:
            tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299)), add_channels()])
        else :
            tx = transforms.Compose([transforms.Resize((299,299)), add_channels()])
        t,s = self.getDataLoaders(batch_size=batchsize,shuffle = True, device=device, transform=tx)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        print(nb_batches)
        m1, s1 = calculate_activation_statistics(s, model, dims, device=device, nb_batches = nb_batches)

        # Compare with joint generations
        data = self.generate(N = batchsize*nb_batches)
        data = torch.stack(data)
        tx = transforms.Compose([ transforms.Resize((299,299)), add_channels()])
        dataset = MultimodalBasicDataset(data, tx)
        gen_dataloader = DataLoader(dataset,batch_size=batchsize, shuffle = True)

        m2, s2 = calculate_activation_statistics(gen_dataloader, model, dims, device=device, nb_batches=nb_batches)
        return calculate_frechet_distance(m1[:dims], s1[:dims, :dims], m2[:dims], s2[:dims, :dims])

    def compute_metrics(self, epoch):
        if epoch%10 != 1:
            return {}
        fid = self.compute_fid(batchsize=64, device='cuda',dims=2048, nb_batches=20)
        return {'fid' : fid}

