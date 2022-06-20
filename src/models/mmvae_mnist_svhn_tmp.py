# MNIST-SVHN multi-modal model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid
from dataloaders import MNIST_SVHN_DL
from utils import update_details
from vis import plot_embeddings, plot_kls_df
from .mmvae_tmp import MMVAE
from .vae_mnist import MNIST
from .vae_svhn import SVHN
from analysis import MnistClassifier, SVHNClassifier
from torchvision import transforms


# Define the classifiers for analysis
classifier1, classifier2 = MnistClassifier(), SVHNClassifier()
path1 = '../experiments/classifier_numbers/2022-06-09/model_4.pt'
path2 = '../experiments/classifier_svhn/2022-06-16/model_8.pt'
classifier1.load_state_dict(torch.load(path1))
classifier2.load_state_dict(torch.load(path2))
# Set in eval mode
classifier1.eval()
classifier2.eval()
# Set to cuda
classifier1.cuda()
classifier2.cuda()



class MNIST_SVHN(MMVAE):
    def __init__(self, params):
        super(MNIST_SVHN, self).__init__(params, MNIST, SVHN)
        self.modelName = 'mnist-svhn'
        self.data_path = params.data_path
        device = 'cuda' if params.cuda else 'cpu'
        self._pz_params = torch.zeros(params.latent_dim).to(device),torch.ones(params.latent_dim).to(device)
        print(self.parameters())

    @property
    def pz_params(self):
        return self._pz_params

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform = transforms.ToTensor()):
        return MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)


    def reconstruct(self, data, runPath, epoch):
        return
        # recons_mat = super(MNIST_SVHN, self).reconstruct([d[:8] for d in data])
        # for r, recons_list in enumerate(recons_mat):
        #     for o, recon in enumerate(recons_list):
        #         _data = data[r][:8].cpu()
        #         recon = recon.squeeze(0).cpu()
        #         # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
        #         _data = _data if r == 1 else resize_img(_data, self.vaes[1].dataSize)
        #         recon = recon if o == 1 else resize_img(recon, self.vaes[1].dataSize)
        #         comp = torch.cat([_data, recon])
        #         save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch, ticks=None):
        return
        # zemb, zsl, kls_df = super(MNIST_SVHN, self).analyse(data, K=10)
        # labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        # plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch), ticks = ticks, K=10)
        # plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))


    def conditional_labels(self, data, n_data=8, ns=30):
        return
        # """ Sample ns from the conditional distribution (for each of the first n_data)
        # and compute the labels repartition in this conditional distribution (based on the
        # predefined classifiers)"""
        #
        # bdata = [d[:n_data] for d in data]
        # samples = self.sample_from_conditional(self, bdata, n=ns)
        # cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]
        #
        # # Compute the labels
        # preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, 3, 32, 32))  # 8*n x 10
        # labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)
        #
        # preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, 1, 28, 28))  # 8*n x 10
        # labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=30):
        return
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        # # Compute general metrics (FID)
        # general_metrics = MMVAE.compute_metrics(self,data, runPath,epoch,freq=10, to_tensor=True)
        #
        # # Compute cross_coherence
        # labels2, labels1 = self.conditional_labels(data, n_data, ns)
        # # Create an extended classes array where each original label is replicated ns times
        # classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        # acc2 = torch.sum(classes_mul == labels2)/(n_data*ns)
        # acc1 = torch.sum(classes_mul == labels1)/(n_data*ns)
        #
        # metrics = dict(accuracy1 = acc1, accuracy2 = acc2)
        #
        # # Compute joint-coherence
        # data = self.generate(runPath, epoch, N=100)
        # labels_mnist = torch.argmax(classifier1(data[0]), dim=1)
        # labels_svhn = torch.argmax(classifier2(data[1]), dim=1)
        #
        # joint_acc = torch.sum(labels_mnist == labels_svhn)/100
        # metrics['joint_coherence'] = joint_acc
        # update_details(metrics, general_metrics)
        #
        # return metrics
        #
        #
        #
def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
