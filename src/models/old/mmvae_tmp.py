# Base MMVAE class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
import wandb

from utils import get_mean, kl_divergence, adjust_shape, add_channels
from vis import embed_umap, tensors_to_df, save_samples
from torchvision import transforms
from dataloaders import MultimodalBasicDataset
from analysis.pytorch_fid import get_activations,calculate_activation_statistics,calculate_frechet_distance
from analysis.pytorch_fid.inception import InceptionV3


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}


class MMVAE(nn.Module):
    def __init__(self, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = dist_dict[params.dist]
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pz_params = None  # defined in subclass
        self.align = -1

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            # encode each modality with its specific encoder
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        print(qz_x, px_zs, zss)
        1/0
        return qz_xs, px_zs, zss

    def generate(self, runPath, epoch, N=8, save=False):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))

        if save:
            data = [*adjust_shape(data[0], data[1])]
            save_samples(data, '{}/generate_{:03d}.png'.format(runPath, epoch))
            wandb.log({'generate_joint': wandb.Image('{}/generate_{:03d}.png'.format(runPath, epoch))})

        return data  # list of generations---one for each modality

    def generate_from_conditional(self,runPath, epoch, N=10, save=False):
        """ Generate samples using the bayes formula : p(x,y) = p(x)p(y|x)"""

        # First step : generate samples from prior --> p(x), p(y)
        data = self.generate(runPath, epoch, N=N)

        # Second step : generate one modality from the other --> p(x|y) and p(y|x)
        cond_data = self.sample_from_conditional(data,n=1)

        # Rearrange the data, only keep the cross modal generations
        reorganized = [[*adjust_shape(data[0],torch.cat(cond_data[0][1]))], [*adjust_shape(torch.cat(cond_data[1][0]), data[1])]]
        if save:
            save_samples(reorganized[0], '{}/gen_from_cond_0_{:03d}.png'.format(runPath, epoch))
            wandb.log({'gen_from_cond_0' : wandb.Image('{}/gen_from_cond_0_{:03d}.png'.format(runPath, epoch))})
            save_samples(reorganized[1], '{}/gen_from_cond_1_{:03d}.png'.format(runPath, epoch))
            wandb.log({'gen_from_cond_1' : wandb.Image('{}/gen_from_cond_1_{:03d}.png'.format(runPath, epoch))})

        return reorganized

    def reconstruct(self, data):
        self.eval()
        return
        # with torch.no_grad():
        #     _, px_zs, _ = self.forward(data)
        #     # cross-modal matrix of reconstructions : reconstruction of modality 1 given modality 2 / given modality 1 etc...
        #     recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        # return recons

    def analyse(self, data, K):
        """Compute all embeddings, and plot empirical posterior and prior distribution.
        It also computes KL distances between distributions"""
        self.eval()
        return
        # with torch.no_grad():
        #     qz_xs, _, zss = self.forward(data, K=K)
        #     pz = self.pz(*self.pz_params) # prior
        #
        #     # Add prior samples to the samples from each encoder generated during the forward pass
        #     # zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
        #     #        *[zs.permute(1,0,2).reshape(-1, zs.size(-1)) for zs in zss]]
        #
        #     zss = [*[zs.permute(1,0,2).reshape(-1, zs.size(-1)) for zs in zss]] # No prior samples
        #     # Labels
        #     zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
        #     kls_df = tensors_to_df(
        #         [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
        #          *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
        #            for p, q in combinations(qz_xs, 2)]],
        #         head='KL',
        #         keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
        #               *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
        #                 for i, j in combinations(range(len(qz_xs)), 2)]],
        #         ax_names=['Dimensions', r'KL$(q\,||\,p)$']
        #     )
        #
        # return torch.cat(zss, 0).cpu().numpy(), \
        #     torch.cat(zsl, 0).cpu().numpy(), \
        #     kls_df

            # previously embed_umap(torch.cat(zss, 0).cpu().numpy()) but incompatibility with u_map version

    def analyse_posterior(self,data, n_samples=8):
        """ For all points in data, computes the mean and standard of q(z|x), q(z|y)"""
        bdata = [d[:n_samples] for d in data]
        qz_xs, _, _ = self.forward(bdata)
        means, std = [], []
        for m, qz_x in enumerate(qz_xs):
            means.append(get_mean(qz_x))
            std.append(qz_x.stddev)
        return means, std

    def cross_modalities_sample_unaligned(self, data, n = 10):
        """ If not all sample dimensions are shared, we need a different
        cross modality generation where the shared variables are generated using
        q(z|xi) and the rest are sampled from the prior"""
        m = len(data) # nb_modalities
        samples = [[None for _ in range(m)] for _ in range(m)]
        for i in range(m):
            for j in range(m):

                zss = self.vaes[i].forward(data[i], K=n)[2]
                pz = self.pz(*self.pz_params)
                z_prior = pz.rsample(zss.size()[:-1]).squeeze()
                zss[:,:,self.align:] = z_prior[:,:,self.align:] if i!=j else zss[:,:,self.align:]
                samples[i][j] = self.vaes[j].dec(zss)[0] # shape [n,n_data,1,32,32]

        return samples






    def sample_from_conditional(self,data, runPath, epoch,n = 10):
        """output recons is a tensor with shape n_mod x n_mod x n x ch x w xh"""

        self.eval()
        return
        # px_zss = [self.forward(data)[1] for _ in range(n)] # sample from qz_xs
        # if self.align != -1:
        #     return self.cross_modalities_sample_unaligned(data, n)
        # with torch.no_grad():
        #     # cross-modal matrix of reconstructions : reconstruction of modality 1 given modality 2 / given modality 1 etc...
        #     recons = [torch.stack([torch.stack([get_mean(px_z) for px_z in r]) for r in px_zs]) for px_zs in px_zss]
        #     recons = torch.stack(recons).squeeze().permute(1,2,0,3,4,5)
        #     print(recons.shape)
        # return recons

    def compute_fid(self,gen_data, batchsize, device, dims=2048, nb_batches=20, to_tensor=False, compare_with='joint'):
        return
        # if to_tensor:
        #     tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299)), add_channels()])
        # else :
        #     tx = transforms.Compose([transforms.Resize((299,299)), add_channels()])
        # t,s = self.getDataLoaders(batch_size=batchsize,shuffle = True, device=device, transform=tx)
        #
        # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        # model = InceptionV3([block_idx]).to(device)
        # m1, s1 = calculate_activation_statistics(s, model, dims, device=device, nb_batches = nb_batches)
        #
        # # _,gen_dataloader = self.getDataLoaders(batch_size=batchsize,shuffle = True, device=device, transform=tx, random=True)
        # #
        # data = torch.stack(adjust_shape(gen_data[0], gen_data[1]))
        # tx = transforms.Compose([ transforms.Resize((299,299)), add_channels()])
        # dataset = MultimodalBasicDataset(data, tx)
        # gen_dataloader = DataLoader(dataset,batch_size=batchsize, shuffle = True)
        #
        # m2, s2 = calculate_activation_statistics(gen_dataloader, model, dims, device=device, nb_batches=nb_batches)
        # return  calculate_frechet_distance(m1, s1, m2, s2)
        #

    def compute_metrics(self,data, runPath, epoch, freq=5, to_tensor=False):
        return
        # print(epoch)
        # if epoch % freq != 1:
        #     return {}
        # batchsize, nb_batches = 64, 100
        # fids = {}
        # if epoch <= (self.params.warmup // freq + 1) * freq:  # Compute fid between joint generation and test set
        #     gen_data = self.generate(runPath, epoch, N=batchsize * nb_batches)
        #     fid = self.compute_fid(gen_data, batchsize, device='cuda', dims=2048, to_tensor=to_tensor,
        #                            nb_batches=nb_batches)
        #     fids['fid_joint'] = fid
        #
        # # Compute fid between test set and joint distributions computed from conditional
        # if epoch >= self.params.warmup:
        #     cond_gen_data = self.generate_from_conditional(runPath, epoch, N=batchsize * nb_batches)
        #     for i, gen_data in enumerate(cond_gen_data):
        #         fids[f'fids_{i}'] = self.compute_fid(gen_data, batchsize, device='cuda', dims=2048, to_tensor=to_tensor,
        #                                              nb_batches=nb_batches)
        #
        # return fids