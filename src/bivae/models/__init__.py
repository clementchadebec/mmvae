
from .mmvae import  MNIST_SVHN as VAE_mnist_svhn
from .mmvae import MMVAE_CIRCLES as VAE_circles_squares
# from .mmvae_cercles_discs import CIRCLES_DISCS as VAE_circles_discs
# from .mmvae_mnist_fashion import MNIST_FASHION as VAE_mnist_fashion
# from .vae_cub_image import CUB_Image as VAE_cubI
# from .vae_cub_image_ft import CUB_Image_ft as VAE_cubIft
# from .vae_cub_sent import CUB_Sentence as VAE_cubS
# from .vae_mnist import MNIST as VAE_mnist
# from .vae_svhn import SVHN as VAE_svhn
# from .j_mnist_fashion import J_MNIST_FASHION as VAE_j_mnist_fashion
# from .j_circles_discs import J_CIRCLES_DISCS as VAE_j_circles_discs
from .jmvae_nf import JMVAE_NF_CIRCLES as VAE_jnf_circles_squares
from .jmvae_nf import JMVAE_NF_MNIST as VAE_jnf_mnist_fashion
from .multi_vaes import Multi_VAES
from .mmvae import MMVAE_MNIST as VAE_mnist_fashion
from .jmvae_nf import JMVAE_NF_DCCA_MNIST_SVHN as VAE_jnf_mnist_svhn_dcca
from .jmvae_nf import JMVAE_NF_MNIST_CONTOUR as VAE_jnf_mnist_contour
from .jmvae_nf import JMVAE_NF_CELEBA as VAE_jnf_celeba
from .mmvae.mmvae_celeba import celeba as VAE_mmvae_celeba
from .mmvae_nf.mnist_svhn import MNIST_SVHN as VAE_mmvae_nf_mnist_svhn
from .mvae.mnist_svhn import MNIST_SVHN as VAE_mvae_mnist_svhn
from .moepoe.mnist_svhn import MNIST_SVHN as VAE_moepoe_mnist_svhn
from .mvae.celeba import celeba as VAE_mvae_celeba
from .moepoe.celeba import celeba as VAE_moepoe_celeba
from .mmvae_nf.celeba import celeba as VAE_mmvae_nf_celeba
from .jmvae_nf.mnist_svhn_fashion import MNIST_SVHN_FASHION as VAE_jnf_msf
from .mmvae.mnist_svhn_fashion import MNIST_SVHN_FASHION as VAE_mmvae_msf
from .mvae.msf import MNIST_SVHN_FASHION as VAE_mvae_msf
from .jmvae_nf.medmnist import MEDMNIST as VAE_jnf_medmnist
from .mmvae.medmnist import MEDMNIST as VAE_mmvae_medmnist
from .mvae.medmnist import MEDMNIST as VAE_mvae_medmnist
from .jmvae_nf.chest_svhn import CHEST_SVHN as VAE_jnf_chest_svhn

__all__ = [ 'VAE_mnist_svhn',
            'VAE_circles_squares',
            'VAE_jnf_circles_squares',
            'VAE_jnf_mnist_fashion',
            'VAE_mnist_fashion',
            'VAE_jnf_mnist_svhn_dcca',
            'VAE_jnf_mnist_contour',
            'VAE_jnf_celeba',
            'VAE_mmvae_celeba',
            'VAE_mmvae_nf_mnist_svhn',
            'VAE_mvae_mnist_svhn',
            'VAE_moepoe_mnist_svhn',
            'VAE_mvae_celeba',
            'VAE_moepoe_celeba',
            'VAE_mmvae_nf_celeba', 
            'VAE_jnf_msf',
            'VAE_mmvae_msf', 
            'VAE_mvae_msf',
            'VAE_jnf_medmnist', 
            'VAE_mmvae_medmnist',
            'VAE_mvae_medmnist',
            'VAE_jnf_chest_svhn'

            ]