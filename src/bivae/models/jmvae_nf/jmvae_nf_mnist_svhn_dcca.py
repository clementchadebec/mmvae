# JMVAE_NF specification for MNIST-SVHN experiment --> Using DCCA to extract shared information

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn
from torchvision.utils import save_image
import pythae
from bivae.my_pythae.models import my_VAE_LinNF, VAE_LinNF_Config, my_VAE_IAF, VAE_IAF_Config, my_VAE, VAEConfig
from bivae.my_pythae.models.nn import Encoder_VAE_MLP, Decoder_AE_MLP, Encoder_VAE_SVHN
from torchnet.dataset import TensorDataset
from torch.utils.data import DataLoader
from bivae.utils import extract_rayon
from bivae.dataloaders import MNIST_SVHN_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Decoder_VAE_SVHN, TwoStepsDecoder, TwoStepsEncoder
import torch.nn.functional as F

from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import MnistClassifier, SVHNClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}

hidden_dim = 512
pretrained_vaes = False

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

# pre_trained encoders as heads for the joint encoder
pretrained_encoders, pretrained_decoders = [None, None], [None, None]
pre_configs = [VAEConfig((1,28,28), 20), VAEConfig((3,32,32), 20)]

dcca_encoders = [torch.load('../dcca/mnist_svhnmodel1.pt'), torch.load('../dcca/mnist_svhnmodel2.pt')]
dcca_configs = [VAEConfig((1,28,28), 15), VAEConfig((3,32,32), 15)]

class JMVAE_NF_DCCA_MNIST_SVHN(JMVAE_NF):
    def __init__(self, params):
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig

        # Define the joint encoder
        joint_encoder = DoubleHeadJoint(hidden_dim,pre_configs[0], pre_configs[1],
                                        Encoder_VAE_MLP ,
                                        Encoder_VAE_SVHN,
                                        params,
                                        pretrained_encoders)

        vae = my_VAE_IAF if not params.no_nf else my_VAE

        # Define the unimodal encoders

        # First load the DCCA encoders
        vae_config1 = vae_config((1, 28, 28), params.latent_dim)
        vae_config2 = vae_config((3, 32, 32), params.latent_dim)
        self.dcca = [Encoder_VAE_MLP(dcca_configs[0]), Encoder_VAE_SVHN(dcca_configs[1])]
        self.dcca[0].load_state_dict(dcca_encoders[0])
        self.dcca[1].load_state_dict(dcca_encoders[1])

        # Then design
        encoder1 = TwoStepsEncoder(self.dcca[0], params)
        encoder2 = TwoStepsEncoder(self.dcca[1], params)

        # Define the decoders
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        # decoder1 = TwoStepsDecoder(Decoder_AE_MNIST,pre_configs[0], pretrained_decoders[0], params)
        # decoder2 = TwoStepsDecoder(Decoder_VAE_SVHN, pre_configs[1], pretrained_decoders[1], params)
        # decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)
        ])

        super(JMVAE_NF_DCCA_MNIST_SVHN, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_dcca_mnist_svhn'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.lik_scaling = ((3*32*32)/(1*28*28), 1) if params.llik_scaling == 0.0 else (params.llik_scaling, 1)
        self.to_tensor = True

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def conditional_labels(self, data, n_data=8, ns=30):
        """ Sample ns from the conditional distribution (for each of the first n_data)
        and compute the labels repartition in this conditional distribution (based on the
        predefined classifiers)"""

        bdata = [d[:n_data] for d in data]
        samples = JMVAE_NF._sample_from_conditional(self, bdata, n=ns)
        cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

        # Compute the labels
        preds2 = classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, 3, 32, 32))  # 8*n x 10
        labels2 = torch.argmax(preds2, dim=1).reshape(n_data, ns)

        preds1 = classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, 1, 28, 28))  # 8*n x 10
        labels1 = torch.argmax(preds1, dim=1).reshape(n_data, ns)

        return labels2, labels1





    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=100, freq=10):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        # Compute general metrics (FID)
        general_metrics = JMVAE_NF.compute_metrics(self,runPath,epoch,freq=freq)

        # Compute cross_coherence
        labels2, labels1 = self.conditional_labels(data, n_data, ns)
        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        acc2 = torch.sum(classes_mul == labels2)/(n_data*ns)
        acc1 = torch.sum(classes_mul == labels1)/(n_data*ns)

        metrics = dict(accuracy1 = acc1, accuracy2 = acc2)
        data = self.generate(runPath, epoch, N=100)
        labels_mnist = torch.argmax(classifier1(data[0]), dim=1)
        labels_svhn = torch.argmax(classifier2(data[1]), dim=1)

        joint_acc = torch.sum(labels_mnist == labels_svhn) / 100
        metrics['joint_coherence'] = joint_acc
        update_details(metrics, general_metrics)

        return metrics

    def compute_recon_loss(self,x,recon,m):
        """Change the way we compute the reocnstruction, through the filter of DCCA"""
        t = self.dcca[m](x).embedding
        recon_t = self.dcca[m](recon).embedding
        return F.mse_loss(t,recon_t,reduction='sum')















