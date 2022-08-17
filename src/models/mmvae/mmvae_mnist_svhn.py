# JMVAE_NF specification for MNIST-SVHN experiment

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms

from utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn
from torchvision.utils import save_image
from pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config, my_VAE, VAEConfig
from pythae.models.nn import Encoder_VAE_MLP, Decoder_AE_MLP
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from utils import extract_rayon
from dataloaders import MNIST_SVHN_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Encoder_VAE_SVHN, Decoder_VAE_SVHN


from ..nn import DoubleHeadMLP, DoubleHeadJoint
from .mmvae import MMVAE
from analysis import MnistClassifier, SVHNClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

hidden_dim = 512

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
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae = my_VAE


        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) # Standard MLP for
        # encoder1, encoder2 = None, None
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        # decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(MNIST_SVHN, self).__init__(params, vaes)
        self.modelName = 'mmvae_mnist_svhn'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1) if params.llik_scaling == 0 else (params.llik_scaling, 1)
        self.to_tensor = True

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def conditional_labels(self, data, n_data=8, ns=30):
        """ Sample ns from the conditional distribution (for each of the first n_data)
        and compute the labels repartition in this conditional distribution (based on the
        predefined classifiers)"""

        bdata = [d[:n_data] for d in data]
        samples = self._sample_from_conditional( bdata, n=ns)
        cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

        # Compute the labels
        preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, 3, 32, 32))  # 8*n x 10
        labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

        preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, 1, 28, 28))  # 8*n x 10
        labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

        return labels2, labels1

    def conditional_dist(self, data, runPath, epoch, n=20):
        """ Plot the conditional distribution of the labels that was computed with
        conditional labels """
        hist_values = torch.cat(self.extract_hist_values(data), dim=0)

        plot_hist(hist_values, '{}/hist_{:03d}.png'.format(runPath, epoch), range=(0, 10))
        wandb.log({'histograms' : wandb.Image('{}/hist_{:03d}.png'.format(runPath, epoch))})



    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=100, freq=10):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        # Compute general metrics (FID)
        general_metrics = MMVAE.compute_metrics(self,runPath,epoch,freq=freq, to_tensor=True)
        # general_metrics = {}
        # Compute cross_coherence
        labels2, labels1 = self.conditional_labels(data, n_data, ns)
        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        acc2 = torch.sum(classes_mul == labels2)/(n_data*ns)
        acc1 = torch.sum(classes_mul == labels1)/(n_data*ns)

        metrics = dict(accuracy1 = acc1, accuracy2 = acc2)

        # Compute joint-coherence
        data = self.generate(runPath, epoch, N=100)
        labels_mnist = torch.argmax(classifier1(data[0]), dim=1)
        labels_svhn = torch.argmax(classifier2(data[1]), dim=1)

        joint_acc = torch.sum(labels_mnist == labels_svhn)/100
        metrics['joint_coherence'] = joint_acc
        update_details(metrics, general_metrics)

        return metrics









