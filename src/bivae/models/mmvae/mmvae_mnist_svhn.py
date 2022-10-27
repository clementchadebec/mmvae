# JMVAE_NF specification for MNIST-SVHN experiment

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import wandb
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from torchvision import transforms

from bivae.analysis.classifiers import load_pretrained_mnist, load_pretrained_svhn
from bivae.dataloaders import MNIST_SVHN_DL, BINARY_MNIST_SVHN_DL
from bivae.my_pythae.models import my_VAE, laplace_VAE
from bivae.utils import update_details
from bivae.vis import plot_hist
from .mmvae import MMVAE
from ..nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.utils import add_channels, unpack_data
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

hidden_dim = 512




class MNIST_SVHN(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE


        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) # Standard MLP for
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)

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

    def set_classifiers(self):
        
        self.classifier1 = load_pretrained_mnist()
        self.classifier2 = load_pretrained_svhn()
        


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
        preds2 = self.classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, 3, 32, 32))  # 8*n x 10
        labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

        preds1 = self.classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, 1, 28, 28))  # 8*n x 10
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
        general_metrics = MMVAE.compute_metrics(self,runPath,epoch,freq=freq)
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
        labels_mnist = torch.argmax(self.classifier1(data[0]), dim=1)
        labels_svhn = torch.argmax(self.classifier2(data[1]), dim=1)

        joint_acc = torch.sum(labels_mnist == labels_svhn)/100
        metrics['joint_coherence'] = joint_acc
        update_details(metrics, general_metrics)

        return metrics

    def compute_fid(self, batch_size):

        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = self.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[],[]]

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations[0].append(model(data[0]))
            ref_activations[1].append(model(data[1]))

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = self.getDataLoaders(batch_size)

        gen_samples = [[],[]]
        for dataT in test:
            data = unpack_data(dataT)
            gen = self._sample_from_conditional(data, n=1)
            gen_samples[0].extend(gen[1][0])
            gen_samples[1].extend(gen[0][1])

        gen_samples = [torch.cat(g).squeeze(0) for g in gen_samples]
        print(gen_samples[0].shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_dataset = MultimodalBasicDataset(gen_samples, tx)
        gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size)

        gen_activations = [[],[]]
        for dataT in gen_dataloader:
            data = unpack_data(dataT)
            gen_activations[0].append(model(data[0]))
            gen_activations[1].append(model(data[1]))
        gen_activations = [np.concatenate(g) for g in gen_activations]

        cond_fids = {}
        for i in range(len(ref_activations)):
            mu1, mu2 = np.mean(ref_activations[i], axis=0), np.mean(gen_activations[i], axis=0)
            sigma1, sigma2 = np.cov(ref_activations[i], rowvar=False), np.cov(gen_activations[i], rowvar=False)

            # print(mu1.shape, sigma1.shape)

            cond_fids[f'fid_{i}'] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return cond_fids








