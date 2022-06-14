# JMVAE_NF specification for MNIST

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb

from utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config, my_VAE, VAEConfig
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from utils import extract_rayon
from ..dataloaders import MNIST_FASHION_DATALOADER
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST


from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP, DoubleHeadMnist
from ..jmvae_nf import JMVAE_NF
from analysis import MnistClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1, 28, 28)

hidden_dim = 512

# Define the classifiers for analysis
classifier1, classifier2 = MnistClassifier(), MnistClassifier()
path1 = '../experiments/classifier_numbers/2022-06-09/model_4.pt'
path2 = '../experiments/classifier_fashion/2022-06-09/model_4.pt'
classifier1.load_state_dict(torch.load(path1))
classifier2.load_state_dict(torch.load(path2))
# Set in eval mode
classifier1.eval()
classifier2.eval()
# Set to cuda
classifier1.cuda()
classifier2.cuda()


class JMVAE_NF_MNIST(JMVAE_NF):
    def __init__(self, params):
        # joint_encoder = DoubleHeadMnist(hidden_dim, params.num_hidden_layers,params)
        joint_encoder = DoubleHeadMLP(28 * 28, 28 * 28, hidden_dim, params.latent_dim, 1)
        vae = my_VAE_IAF if not params.no_nf else my_VAE
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig
        vae_config = vae_config(input_dim, params.latent_dim)

        # encoder1, encoder2 = Encoder_VAE_MNIST(vae_config), Encoder_VAE_MNIST(vae_config)
        encoder1, encoder2 = None, None
        # decoder1, decoder2 = Decoder_AE_MNIST(vae_config), Decoder_AE_MNIST(vae_config)
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config, encoder=encoder2, decoder=decoder2)

        ])
        super(JMVAE_NF_MNIST, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_mnist'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'fashion'

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = None):
        train, test = MNIST_FASHION_DATALOADER(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        JMVAE_NF.sample_from_conditional(self, data, runPath, epoch, n)

    def extract_hist_values(self, data, n_data = 8, ns = 30):
        bdata = [d[:n_data] for d in data]
        samples = JMVAE_NF._sample_from_conditional(self, bdata, n=ns)
        cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

        # Compute the labels
        preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data*ns, 1, 28, 28))  # 8*n x 10
        labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

        preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data*ns, 1, 28, 28))  # 8*n x 10
        labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

        return labels2, labels1

    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=30):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""


        labels2, labels1 = self.extract_hist_values(data, n_data, ns)
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        good_samples = torch.div(labels2 - 1,3, rounding_mode='trunc') == classes_mul
        acc2 = torch.sum(good_samples)/(n_data*ns)
        acc1 = torch.sum(labels1 == classes_mul)/(n_data*ns)

        # Compute entropy of the good_samples
        labels2_acc = good_samples*labels2
        neg_entrop = negative_entropy(labels2_acc.cpu(),range=(0,10), bins=10)

        metrics = dict(acc2=acc2, acc1 =acc1, neg_entropy = neg_entrop)
        general_metrics = JMVAE_NF.compute_metrics(self,epoch, to_tensor=True)
        update_details(metrics, general_metrics)
        return metrics


    def conditional_dist(self, data, runPath, epoch, n=20):
        # bdata = [d[:8] for d in data]
        # samples = JMVAE_NF._sample_from_conditional(self, bdata, n=n)  # sample[0][1] is of shape n x 8 x ch x w x h
        hist_values = torch.cat(self.extract_hist_values(data), dim=0)

        plot_hist(hist_values, '{}/hist_{:03d}.png'.format(runPath, epoch), range=(0, 10))
        wandb.log({'histograms' : wandb.Image('{}/hist_{:03d}.png'.format(runPath, epoch))})

    def analyse(self, data, runPath, epoch, classes=None):
        # Visualize the joint latent space
        m, s, zxy = self.analyse_joint_posterior(data, n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))

        if self.params.latent_dim > 2:
            zxy = TSNE().fit_transform(zxy)
            zx = TSNE().fit_transform(zx)
            zy = TSNE().fit_transform(zy)

        plot_embeddings_colorbars(zxy, zxy, classes[0], classes[1], "{}/joint_embedding_{:03d}.png".format(runPath, epoch))
        wandb.log({'joint_embedding' : wandb.Image("{}/joint_embedding_{:03d}.png".format(runPath, epoch))})
        plot_embeddings_colorbars(zx,zy,classes[0], classes[1], "{}/uni_embedding_{:03d}.png".format(runPath,epoch))
        wandb.log({'uni_embedding' : wandb.Image("{}/uni_embedding_{:03d}.png".format(runPath,epoch))})
        # Analyse histograms of conditional samples
        super().analyse(data, runPath, epoch)
        self.conditional_dist(data, runPath, epoch, n=100)








