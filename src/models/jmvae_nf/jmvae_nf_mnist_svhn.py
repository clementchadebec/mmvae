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
from ..dataloaders import MNIST_SVHN_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Encoder_VAE_SVHN, Decoder_VAE_SVHN


from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from analysis import MnistClassifier

dist_dict = {'normal': dist.Normal, 'laplace': dist.Laplace}
input_dim = (1, 28, 28)

hidden_dim = 512

# Define the classifiers for analysis
# classifier1, classifier2 = MnistClassifier(), MnistClassifier()
# path1 = '../experiments/classifier_numbers/2022-06-09/model_4.pt'
# path2 = '../experiments/classifier_fashion/2022-06-09/model_4.pt'
# classifier1.load_state_dict(torch.load(path1))
# classifier2.load_state_dict(torch.load(path2))
# # Set in eval mode
# classifier1.eval()
# classifier2.eval()
# # Set to cuda
# classifier1.cuda()
# classifier2.cuda()


class JMVAE_NF_MNIST_SVHN(JMVAE_NF):
    def __init__(self, params):
        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig
        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        joint_encoder = DoubleHeadJoint(hidden_dim, params.num_hidden_layers,vae_config1, vae_config2, Encoder_VAE_MLP, Encoder_VAE_SVHN)
        vae = my_VAE_IAF if not params.no_nf else my_VAE


        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2) # Standard MLP for
        # encoder1, encoder2 = None, None
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        # decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(JMVAE_NF_MNIST_SVHN, self).__init__(params, joint_encoder, vaes)
        self.modelName = 'jmvae_nf_mnist_svhn'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'


    # def generate(self, runPath, epoch, N=8, save=False):
    #     data = JMVAE_NF.generate(self, runPath, epoch, N=N)
    #     if save :
    #         save_samples_mnist_svhn(data, '{}/generate_{:03d}.png'.format(runPath, epoch))
    #         wandb.log({'generate_joint': wandb.Image('{}/generate_{:03d}.png'.format(runPath, epoch))})
    #     return data

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test



    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=30):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        metrics = JMVAE_NF.compute_metrics(self,runPath,epoch, to_tensor=True)
        # metrics = {}
        return metrics


    def analyse(self, data, runPath, epoch, classes=None):
        # Visualize the joint latent space
        m, s, zxy = self.analyse_joint_posterior(data, n_samples=len(data[0]))
        zx, zy = self.analyse_uni_posterior(data,n_samples=len(data[0]))

        if self.params.latent_dim > 2:
            zxy = TSNE().fit_transform(zxy)
            zx = TSNE().fit_transform(zx)
            zy = TSNE().fit_transform(zy)

        plot_embeddings_colorbars(zxy, zxy, classes[0], classes[1], "{}/joint_embedding_{:03d}.png".format(runPath, epoch), ax_lim=None)
        wandb.log({'joint_embedding' : wandb.Image("{}/joint_embedding_{:03d}.png".format(runPath, epoch))})
        plot_embeddings_colorbars(zx,zy,classes[0], classes[1], "{}/uni_embedding_{:03d}.png".format(runPath,epoch), ax_lim = None)
        wandb.log({'uni_embedding' : wandb.Image("{}/uni_embedding_{:03d}.png".format(runPath,epoch))})










