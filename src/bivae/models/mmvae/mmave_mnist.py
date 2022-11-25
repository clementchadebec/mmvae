
from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist
from torchvision.utils import save_image
from pythae.models import VAE_IAF_Config, VAE_LinNF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_IAF

from bivae.dataloaders import MNIST_FASHION_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST


from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP
from ..mmvae import MMVAE
from bivae.analysis.classifiers.classifier_mnist import load_pretrained_fashion, load_pretrained_mnist

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1, 28, 28)

hidden_dim = 512



def fashion_labels_to_mnist(f_labels):
    return torch.div(f_labels - 1,3, rounding_mode='trunc')


class MMVAE_MNIST(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae_config1 = vae_config((1, 28, 28), params.latent_dim)
        vae_config2 = vae_config((1, 28, 28), params.latent_dim)
        vae = my_VAE

        encoder1, encoder2 = None, None
        decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(MMVAE_MNIST, self).__init__(params, vaes)
        self.modelName = 'mmvae_mnist_fashion'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'fashion'
        self.lik_scaling = (1, 1) if params.llik_scaling == 0 else (params.llik_scaling, 1)

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = None):
            val: object
            train, test, val = MNIST_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
            return train, test, val

    def set_classifiers(self):
        self.classifier1 = load_pretrained_mnist()
        self.classifier2 = load_pretrained_mnist()

    def conditional_labels(self, data, n_data = 8, ns = 30):
        """ Sample ns from the conditional distribution (for each of the first n_data)
        and compute the labels repartition in this conditional distribution (based on the
        predefined classifiers)"""

        bdata = [d[:n_data] for d in data]
        samples = MMVAE._sample_from_conditional(self, bdata, n=ns)
        cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

        # Compute the labels
        preds2 = self.classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data*ns, 1, 28, 28))  # 8*n x 10
        labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

        preds1 = self.classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data*ns, 1, 28, 28))  # 8*n x 10
        labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

        return labels2, labels1

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""


        labels2, labels1 = self.conditional_labels(data, n_data, ns)

        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        good_samples = fashion_labels_to_mnist(labels2) == classes_mul
        acc2 = torch.sum(good_samples)/(n_data*ns)
        acc1 = torch.sum(labels1 == classes_mul)/(n_data*ns)

        # Compute entropy of the good_samples
        labels2_acc = good_samples*labels2
        neg_entrop = negative_entropy(labels2_acc.cpu(),range=(0,10), bins=10)




        metrics = dict(acc2=acc2, acc1 =acc1, neg_entropy = neg_entrop)
        general_metrics = MMVAE.compute_metrics(self,runPath,epoch, freq=freq)

        # Compute joint coherence :
        data = self.generate(runPath, epoch, N=100)
        labels_mnist = torch.argmax(self.classifier1(data[0]), dim=1)
        labels_fashion = torch.argmax(self.classifier2(data[1]), dim=1)

        joint_acc = torch.sum(labels_mnist == fashion_labels_to_mnist(labels_fashion)) / 100
        metrics['joint_coherence'] = joint_acc

        update_details(metrics, general_metrics)
        return metrics


    def conditional_dist(self, data, runPath, epoch, n=20):
        # bdata = [d[:8] for d in data]
        # samples = JMVAE_NF._sample_from_conditional(self, bdata, n=n)  # sample[0][1] is of shape n x 8 x ch x w x h
        hist_values = torch.cat(self.conditional_labels(data), dim=0)

        plot_hist(hist_values, '{}/hist_{:03d}.png'.format(runPath, epoch), range=(0, 10))
        wandb.log({'histograms' : wandb.Image('{}/hist_{:03d}.png'.format(runPath, epoch))})

    def analyse(self, data, runPath, epoch, classes=[None, None]):
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
        super().analyse(data, runPath, epoch, classes=classes)
        self.conditional_dist(data, runPath, epoch, n=100)








