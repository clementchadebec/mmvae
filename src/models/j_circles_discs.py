''' A joint multimodal autoencoder for mnist-fashion dataset'''
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import Constants, tensor_classes_labels
from torchvision.utils import save_image
import torch
from vis import plot_posteriors, plot_embeddings, plot_embeddings_colorbars

from .mmvae import MMVAE
from .mmvae_mnist_fashion import MNIST_FASHION
from .mmvae_cercles_discs import CIRCLES_DISCS
from .jmvae import JMVAE

# Constants
size1, size2 = 32*32 , 32*32 # modality 1 size, modality 2 size
hidden_dim = 512





# Classes


class J_CIRCLES_DISCS(CIRCLES_DISCS, JMVAE):

    def __init__(self, params):
        CIRCLES_DISCS.__init__(self,params)
        JMVAE.__init__(self,params, Enc(params.latent_dim, params.num_hidden_layers), self.vaes)

    def analyse_posterior(self,data, n_samples,runPath,epoch, ticks=None):
        means, stds = MMVAE.analyse_posterior(self,data, n_samples)
        m, s,_ = JMVAE.analyse_joint_posterior(self,data, n_samples)
        means.append(m)
        stds.append(s)
        plot_posteriors(means, stds, '{}/posteriors_{:03}.png'.format(runPath,epoch),
                        [self.vaes[0].modelName, self.vaes[1].modelName, 'joint'], ticks=ticks)

    def analyse(self, data, runPath, epoch, ticks=None, classes=None):
        zss, zsl,kls_df = MMVAE.analyse(self,data,K=1)
        m,s,zxy = JMVAE.analyse_joint_posterior(self,data,n_samples=len(data[0]))
        zss = np.concatenate([zss,zxy],0)
        zsl = np.concatenate([zsl,np.ones(len(zxy))*2],0)
        labels, labels_class = ['qz_squares', 'qz_circles', 'q_joint'], ['empty', 'full']
        zsl, labels = tensor_classes_labels(zsl,3*list(classes),
                                            labels, labels_class) if classes is not None else (zsl, labels)
        plot_embeddings(zss, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch), ticks=ticks, K=1)

    def analyse_rayons(self,data, r0, r1, runPath, epoch):
        m,s,zxy = JMVAE.analyse_joint_posterior(self,data,n_samples=len(data[0]))
        plot_embeddings_colorbars(zxy,zxy,r0,r1,'{}/embedding_rayon_{:03}.png'.format(runPath,epoch))

    def reconstruct(self, data, runPath, epoch):
        recons = JMVAE.reconstruct_jointly(self,[d[:8] for d in data])
        for m, recon in enumerate(recons):
            _data = data[m][:8].cpu()
            recon = recon.squeeze(0).cpu()
            comp = torch.cat([_data, recon])
            save_image(comp, '{}/recon_{}_{:03d}.png'.format(runPath, m, epoch))
