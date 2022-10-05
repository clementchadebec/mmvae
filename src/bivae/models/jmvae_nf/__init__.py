from .jmvae_nf import JMVAE_NF
from .jmvae_nf_circles import JMVAE_NF_CIRCLES
from .jmvae_nf_mnist import JMVAE_NF_MNIST
from .jmvae_nf_mnist_svhn import JMVAE_NF_MNIST_SVHN
from .jmvae_nf_mnist_svhn_dcca import JMVAE_NF_DCCA_MNIST_SVHN
from .mnist_contour import JMVAE_NF_MNIST_CONTOUR

__all__ = ['JMVAE_NF', 'JMVAE_NF_CIRCLES', 'JMVAE_NF_MNIST',
           'JMVAE_NF_MNIST_SVHN',
           'JMVAE_NF_DCCA_MNIST_SVHN',
           'JMVAE_NF_MNIST_CONTOUR'

           ]