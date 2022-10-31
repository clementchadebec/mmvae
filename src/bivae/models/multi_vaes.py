# Define a super class with common functions and attributes for any multimodal vaes
# with one latent space


# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb

import bivae.analysis.prd as prd
from torchvision import transforms
from umap import UMAP
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from bivae.utils import get_mean, kl_divergence, add_channels, adjust_shape, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples
from torchvision.utils import save_image


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace, 'bernoulli' : dist.Bernoulli}
input_dim = (1,32,32)


reducer = UMAP
# reducer = TSNE

class Multi_VAES(nn.Module):
    def __init__(self,params, vaes):
        super(Multi_VAES, self).__init__()
        self.pz = dist_dict[params.dist]
        self.mod = len(vaes)
        self.vaes = vaes
        self.modelName = None # to be populated in subclasses
        self.params = params
        self.data_path = params.data_path
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(1, params.latent_dim), requires_grad=False)  # logvar
        ])
        self.max_epochs = params.epochs
        self.sampler = None
        self.save_format = '.png'
        self.to_tensor = None # to define in each subclass. It says if the data must be formatted to tensor.
        self.ref_activations = None
        self.eval_mode = False

        self.px_z = [ dist_dict[r] for r in params.recon_losses]
        print(f"Set the decoder distributions to {self.px_z}")


    @property
    def pz_params(self):
        return self._pz_params


    def getDataLoaders(self):
        raise "getDataLoaders class must be defined in the subclasses"
        return

    def forward(self, x):
        raise "forward must be defined in the subclass"
        return

    def reconstruct(self):
        raise "reconstruct must be defined in the subclass"

    def generate(self,runPath, epoch, N= 8, save=False):
        """Generate samples from sampling the prior distribution"""
        self.eval()
        with torch.no_grad():
            data = []
            if self.sampler is None:
                pz = self.pz(*self.pz_params)
                latents = pz.rsample(torch.Size([N])).squeeze()
            else :
                latents = self.sampler.sample(num_samples=N)
            for d, vae in enumerate(self.vaes):
                data.append(vae.decoder(latents)["reconstruction"])

        if save and self.mod == 2:
            data = [*adjust_shape(data[0],data[1])]
            file = ('{}/generate_{:03d}'+self.save_format).format(runPath, epoch)
            save_samples(data,file)
            wandb.log({'generate_joint' : wandb.Image(file)})
        return data  # list of generations---one for each modality

    def generate_from_conditional(self,runPath, epoch, N=10, save=False):
        """ Generate samples using the bayes formula : p(x,y) = p(x)p(y|x).
        First sampling a modality from the prior then the other one from the conditinal distribution"""

        # First step : generate samples from prior --> p(x), p(y)
        data = self.generate(runPath, epoch, N=N)

        # Second step : generate one modality from the other --> p(x|y) and p(y|x)
        cond_data = self._sample_from_conditional(data,n=1)

        # Rearrange the data

        reorganized = [[*adjust_shape(data[0],torch.cat(cond_data[0][1]))], [*adjust_shape(torch.cat(cond_data[1][0]), data[1])]]
        if save:
            file = ('{}/gen_from_cond_0_{:03d}'+self.save_format).format(runPath, epoch)
            save_samples(reorganized[0], file)
            wandb.log({'gen_from_cond_0' : wandb.Image(file)})
            file = ('{}/gen_from_cond_1_{:03d}'+self.save_format).format(runPath, epoch)
            save_samples(reorganized[1],file )
            wandb.log({'gen_from_cond_1' : wandb.Image(file)})

        return reorganized

    def analyse_joint_posterior(self,data, n_samples = 10):
        raise "analyse_joint_posterior must be defined in the subclass"



    def analyse(self, data, runPath, epoch, classes=[None,None]):
        # Visualize the joint latent space
        m, s, zxy = self.analyse_joint_posterior(data, n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data, n_samples=len(data[0]))

        # Fit a classifier on the latent space and see the accuracy
        if self.train_latents is not None:
            latent_acc = self.classify_latent(self.train_latents[0], self.train_latents[1],zxy,classes[0])
            wandb.log({'latent_acc' : latent_acc})

        if self.params.latent_dim > 2:
            zxy = reducer().fit_transform(zxy)
            zx = reducer().fit_transform(zx)
            zy = reducer().fit_transform(zy)

        file = ("{}/joint_embedding_{:03d}" + self.save_format).format(runPath, epoch)
        plot_embeddings_colorbars(zxy, zxy, classes[0], classes[1],file
                                  , ax_lim=None)
        wandb.log({'joint_embedding': wandb.Image(file)})
        file = ("{}/uni_embedding_{:03d}" + self.save_format).format(runPath, epoch)
        plot_embeddings_colorbars(zx, zy, classes[0], classes[1], file,
                                  ax_lim=None)
        wandb.log({'uni_embedding': wandb.Image(file)})




    def classify_latent(self,z_train,t_train,z_test,t_test):
        cl = SGDClassifier(loss='hinge',penalty='l2')
        cl.fit(z_train.cpu(),t_train.cpu())
        y_pred = cl.predict(z_test)
        return accuracy_score(y_pred,t_test)


    def analyse_uni_posterior(self, data, n_samples):
        bdata = [d[:n_samples] for d in data]
        zsamples = [self.vaes[m].forward(bdata[m]).z.cpu() for m in range(self.mod)]
        return zsamples


    def analyse_posterior(self,data, n_samples, runPath, epoch, ticks=None, N= 30):
        """ For all points in data, samples N points from q(z|x) and q(z|y)"""
        bdata = [d[:n_samples] for d in data]
        #zsamples[m] is of size N, n_samples, latent_dim
        zsamples = [torch.stack([self.vaes[m].forward(bdata[m]).__getitem__('z') for _ in range(N)]) for m in range(self.mod)]
        file = ('{}/samplepost_{:03d}' + self.save_format).format(runPath, epoch)
        plot_samples_posteriors(zsamples, file, None)
        wandb.log({'sample_posteriors' : wandb.Image(file)})
        return


    def _sample_from_conditional(self,bdata, n=10):
        """Samples from q(z|x) and reconstruct y and conversely"""
        self.eval()
        samples = [[[] for j in range(self.mod)] for i in range(self.mod)]

        with torch.no_grad():

            for _ in range(n):
                                    
                outputs = [self.vaes[i].forward(bdata[i]) for i in range(self.mod)]
                zs = [o.z for o in outputs]
                for i,o in enumerate(outputs):
                    samples[i][i].append(o.recon_x)
                    for j, vae in enumerate(self.vaes):
                        if i!=j:
                            samples[i][j].append(vae.decoder(zs[i])["reconstruction"])
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
                if _data.shape[1:] != recon.shape[1:]:
                        _data, recon = adjust_shape(_data, recon) # modify the shapes in place to match dimensions

                comp = torch.cat([_data, recon])
                filename = '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch)
                save_image(comp, filename)
                wandb.log({'cond_samples_{}x{}.png'.format(r,o) : wandb.Image(filename)})

    def assess_quality(self, assesser,  runPath=None, epoch=0):

        """Compute the three multimodal versions of FID and PRD : on the joint generation and on the two conditional
                        generation"""
        print("Starting to compute FID, PRD metrics")

        # Compute fid between joint generation and test set
        gen_data = self.generate(runPath, epoch, assesser.n_samples)
        # Save the generated data
        gen_dataloader = assesser.GenerateDataloader(gen_data, assesser.gen_transform)
        rd = assesser.compute_fid_prd(gen_dataloader, runPath, compute_unimodal=True)

        list_prds, prd0, prd1 = [rd['prd_data']], rd['prd_data0'], rd['prd_data1']
        metrics = {'fid_joint': rd['fid'], 'unifid0': rd['fid0'], 'unifid1': rd['fid1']}

        # Compute fid between test set and joint distributions computed from conditional
        cond_gen_data = self.generate_from_conditional(runPath, epoch, N=assesser.n_samples)
        for i, gen_data in enumerate(cond_gen_data):
            gen_dataloader = assesser.GenerateDataloader(gen_data, assesser.gen_transform)
            rd = assesser.compute_fid_prd(gen_dataloader, runPath)
            list_prds.append(rd['prd_data'])
            metrics[f'fids_{i}'] = rd['fid']
            # save concatenation of activations
            np.save(f'{runPath}/concat_activ_{i}.npy', rd['concat_activations'])

        np.save('{}/prd_data_{}.npy'.format(runPath, epoch), list_prds)
        np.save('{}/uniprd_data_{}.npy'.format(runPath, epoch), [prd0, prd1])
        prd.plot(list_prds, out_path='{}/prd_plot_{:03d}.png'.format(runPath, epoch),
                 labels=['Joint', 'Cond on 0', 'Cond on 1'])

        # Plot unimodal prds
        prd.plot([prd0,prd1], out_path='{}/uni_prd_plot.png'.format(runPath), labels=['Modality 0', 'Modality 1'],
                 legend_loc = 'upper right')

        return metrics


    def compute_metrics(self, runPath, epoch, freq = 5, num_clusters = 10):
        return {}


    def compute_uni_ll_from_prior(self, data, mod, K=1000, batch_size_K = 100):

        '''Compute an estimate of ln(p(x)) = ln E_{p(z)}(p(x|z)) with monte carlo sampling'''

        prior = self.pz(*self.pz_params)
        z = prior.rsample([len(data[0]), K]).squeeze(-2) # n_data_points, K, latent_dim

        ll = 0
        for i in range(len(data[0])):
            start_idx, stop_index = 0, batch_size_K
            ln_pxs = []
            while stop_index <= K:

                latents = z[i][start_idx:stop_index]

                # Compute p(x|z) for z in latents and for each modality m
                mus = self.vaes[mod].decoder(latents)['reconstruction']  # (batch_size_K, nb_channels, w, h)
                x_m = data[mod][i]  # (nb_channels, w, h)
                if self.px_z[mod] == dist.Bernoulli:
                    lp = self.px_z[mod](mus).log_prob(x_m).sum(dim=(1, 2, 3))
                else:
                    lp = self.px_z[mod](mus, scale=1).log_prob(x_m).sum(dim=(1, 2, 3))

                ln_pxs.append(torch.logsumexp(lp, dim=0))

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K

            ll += torch.logsumexp(torch.Tensor(ln_pxs), dim=0)

        return {f'uni_from_prior_{mod}': ll / len(data[0])}


    def compute_conditional_likelihood_bis(self,data, cond_mod, gen_mod,K=1000,batch_size_K = 100):

        '''
        Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with the same approximation as in MVAE, JMVAE :

        ln p(x|y) = ln E_{q(z|y)}( p(x,y,z)/q(z|y) ) - ln E_{p(z)}(p(y|z))

        Each term is computed using importance sampling

        '''

        # Compute the first term
        t1 = self.compute_joint_ll_from_uni(data, cond_mod, K, batch_size_K)[f'joint_ll_from_{cond_mod}']
        t2 = self.compute_uni_ll_from_prior(data, cond_mod, K=K, batch_size_K=batch_size_K)[f'uni_from_prior_{cond_mod}']
        # print(t1, t2)
        return {f'conditional_likelihood_bis_{cond_mod}_{gen_mod} ' : t1 - t2}


    def compute_conditional_likelihoods(self, data, K=1000, batch_size_K=100):

        metrics = self.compute_conditional_likelihood_bis(data, 0,1, K, batch_size_K)
        update_details(metrics, self.compute_conditional_likelihood_bis(data, 1, 0,K, batch_size_K))
        return metrics





