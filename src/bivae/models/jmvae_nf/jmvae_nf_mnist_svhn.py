# JMVAE_NF specification for MNIST-SVHN experiment

from itertools import combinations

import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn

from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE, my_VAE_LinNF, my_VAE_IAF
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from bivae.models.nn import Encoder_VAE_SVHN

from bivae.dataloaders import MNIST_SVHN_DL
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Decoder_VAE_SVHN, TwoStepsDecoder, TwoStepsEncoder


from ..vae_circles import CIRCLES
from ..nn import DoubleHeadMLP, DoubleHeadJoint
from ..jmvae_nf import JMVAE_NF
from bivae.analysis import MnistClassifier, SVHNClassifier, compute_accuracies

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


# Load pretrained encoders as heads for the joint_encoder
if pretrained_vaes :
    pretrained_encoders = [ torch.load('../experiments/vae_mnist/encoder.pt'),
                torch.load('../experiments/vae_svhn/encoder.pt')
    ]
    pre_configs = [torch.load('../experiments/vae_mnist/config.pt'), torch.load('../experiments/vae_svhn/config.pt')]
    pretrained_decoders = [ torch.load('../experiments/vae_mnist/decoder.pt'),
                torch.load('../experiments/vae_svhn/decoder.pt')]


# Or use random initialized ones
else :
    pretrained_encoders, pretrained_decoders = [None, None], [None, None]
    pre_configs = [VAEConfig((1,28,28), 20), VAEConfig((3,32,32), 20)]


class JMVAE_NF_MNIST_SVHN(JMVAE_NF):
    def __init__(self, params):

        self.shape_mod1,self.shape_mod2 = (1,28,28),(3,32,32)

        vae_config = VAE_IAF_Config if not params.no_nf else VAEConfig

        # Define the joint encoder
        joint_encoder = DoubleHeadJoint(hidden_dim,pre_configs[0], pre_configs[1],
                                        Encoder_VAE_MLP ,
                                        Encoder_VAE_SVHN,
                                        params,
                                        pretrained_encoders)

        vae = my_VAE_IAF if not params.no_nf else my_VAE

        # Define the unimodal encoders
        vae_config1 = vae_config((1, 28, 28), params.latent_dim)
        vae_config2 = vae_config((3, 32, 32), params.latent_dim)

        encoder1, encoder2 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2)

        # Define the decoders
        decoder1, decoder2 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2)
        # decoder1 = TwoStepsDecoder(Decoder_AE_MNIST,pre_configs[0], pretrained_decoders[0], params)
        # decoder2 = TwoStepsDecoder(Decoder_VAE_SVHN, pre_configs[1], pretrained_decoders[1], params)
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
        self.lik_scaling = ((3*32*32)/(1*28*28), 1) if params.llik_scaling == 0.0 else (params.llik_scaling, 1)
        self.to_tensor = True
        self.classifier1 = classifier1
        self.classifier2 = classifier2

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=100, freq=10):
        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        general_metrics = JMVAE_NF.compute_metrics(self, runPath, epoch, freq=freq)
        accuracies = compute_accuracies(self, classifier1, classifier2, data, classes, n_data, ns)

        update_details(accuracies, general_metrics)
        return accuracies













