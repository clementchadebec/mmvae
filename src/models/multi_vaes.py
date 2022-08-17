# Define a super class with common functions and attributes for any multimodal vaes
# with one latent space


# Base JMVAE-NF class definition

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import wandb
from analysis.pytorch_fid import get_activations,calculate_fid_from_embeddings
from analysis.pytorch_fid.inception import InceptionV3
import analysis.prd as prd
from torchvision import transforms
from dataloaders import MultimodalBasicDataset
from umap import UMAP
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from utils import get_mean, kl_divergence, add_channels, adjust_shape
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples
from torchvision.utils import save_image


dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1,32,32)


reducer = UMAP
# reducer = TSNE

class Multi_VAES(nn.Module):
    def __init__(self,params, vaes):
        super(Multi_VAES, self).__init__()
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
        self.max_epochs=params.epochs
        self.sampler = None
        self.save_format = '.png'
        self.to_tensor = None # to define in each subclass. It says if the data must be formatted to tensor.
        self.ref_activations = None

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

        if save:
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
                if _data.shape[1:] != recon.shape[1:]:
                    _data, recon = adjust_shape(_data, recon) # modify the shapes in place to match dimensions

                comp = torch.cat([_data, recon])
                filename = '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch)
                save_image(comp, filename)
                wandb.log({'cond_samples_{}x{}.png'.format(r,o) : wandb.Image(filename)})

    # def compute_fid(self,gen_data, batchsize, device, dims=2048, nb_batches=20):
    #     """
    #     -- gen_data is a dataloader of paired images
    #     -- batchsize is the size that is used for computing FID embedding (make sure it divides the lenght of the
    #     dataloader to not lose any samples
    #
    #     returns : the fid score computed between gen data and the test data (loaded with model.GetDataloaders)
    #
    #     """
    #     if self.to_tensor:
    #         tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299)), add_channels()])
    #     else :
    #         tx = transforms.Compose([transforms.Resize((299,299)), add_channels()])
    #     t,s,v = self.getDataLoaders(batch_size=batchsize,shuffle = True, device=device, transform=tx)
    #
    #     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    #     model = InceptionV3([block_idx]).to(device)
    #     m1, s1 = calculate_activation_statistics(s, model, dims, device=device, nb_batches = nb_batches)
    #
    #     data = torch.stack(adjust_shape(gen_data[0], gen_data[1]))
    #     tx = transforms.Compose([ transforms.Resize((299,299)), add_channels()])
    #     dataset = MultimodalBasicDataset(data, tx)
    #     gen_dataloader = DataLoader(dataset,batch_size=batchsize, shuffle = True)
    #
    #     m2, s2 = calculate_activation_statistics(gen_dataloader, model, dims, device=device, nb_batches=nb_batches)
    #     return  calculate_frechet_distance(m1, s1, m2, s2)

    def compute_fid_prd(self, gen_data, batchsize, device, dims=2048, nb_batches=20, num_clusters=20):
        """
        Compute the prd data between the gen_data given as input and the reference data that is the test dataloader.
        """

        # Defines the transformations to format the images for the Inception network
        if self.to_tensor:
            tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299)), add_channels()])
        else :
            tx = transforms.Compose([transforms.Resize((299,299)), add_channels()])

        # Load the reference data
        t,s,v = self.getDataLoaders(batch_size=batchsize,shuffle = True, device=device, transform=tx)
        # Create a dataloader with the formatted generated data
        data = torch.stack(adjust_shape(gen_data[0], gen_data[1]))
        tx = transforms.Compose([ transforms.Resize((299,299)), add_channels()])
        dataset = MultimodalBasicDataset(data, tx)
        gen_dataloader = DataLoader(dataset,batch_size=batchsize, shuffle = True)

        # Define the model to compute the embeddings
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

        # Compute embeddings
        ref_act = get_activations(s,model, dims=dims, device=device, nb_batches=nb_batches) # numpy array
        gen_act = get_activations(gen_dataloader, model, dims=dims, device=device, nb_batches=nb_batches)

        # Compute prd
        prd_data = prd.compute_prd_from_embedding(ref_act, gen_act, num_clusters=num_clusters)

        # Compute fid
        fid = calculate_fid_from_embeddings(ref_act,gen_act)
        return  prd.compute_prd_from_embedding(ref_act, gen_act, num_clusters=num_clusters)

    def compute_prd_metrics(self,runPath, epoch):
        """Compute the three multimodal versions of PRD : on the joint generation and on the two conditional
                generation"""
        print("Starting to compute PRD metrics")

        batchsize, nb_batches = 64, 100

        # Compute fid between joint generation and test set
        gen_data = self.generate(runPath, epoch, N=batchsize * nb_batches)
        prd_data = self.compute_prd(gen_data, batchsize, device='cuda', dims=2048,
                               nb_batches=nb_batches)
        prd.plot([prd_data],out_path = '{}/prd_plot_joint_{:03d}.png'.format(runPath, epoch))

        # Compute fid between test set and joint distributions computed from conditional
        cond_gen_data = self.generate_from_conditional(runPath, epoch, N=batchsize * nb_batches)
        for i, gen_data in enumerate(cond_gen_data):
            prd_data = self.compute_prd(gen_data, batchsize, device='cuda', dims=2048,
                                                    nb_batches=nb_batches)
            prd.plot([prd_data], out_path='{}/prd_plot_{}_{:03d}.png'.format(runPath,i, epoch))

        return

    def compute_fids_metrics(self,runPath, epoch):
        """Compute the three multimodal versions of FID : on the joint generation and on the two conditional
        generation"""

        batchsize,nb_batches = 64,100
        metrics = {}

         # Compute fid between joint generation and test set
        gen_data = self.generate(runPath, epoch, N = batchsize*nb_batches)
        fid = self.compute_fid(gen_data,batchsize, device='cuda',dims=2048, nb_batches=nb_batches)
        metrics['fid_joint'] = fid

        # Compute fid between test set and joint distributions computed from conditional
        cond_gen_data = self.generate_from_conditional(runPath, epoch, N = batchsize*nb_batches)
        for i,gen_data in enumerate(cond_gen_data):
            metrics[f'fids_{i}'] = self.compute_fid(gen_data,batchsize,device='cuda',dims=2048, nb_batches=nb_batches)


        return metrics

    def compute_metrics(self, runPath, epoch, freq = 5,):
        if epoch%freq != 0 and epoch !=1:
            return {}
        else :
            dict = self.compute_fids_metrics(runPath, epoch)
            self.compute_prd_metrics(runPath, epoch)
            return dict



