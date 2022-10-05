# from .mmvae_cub_images_sentences import CUB_Image_Sentence as VAE_cubIS
# from .mmvae_cub_images_sentences_ft import CUB_Image_Sentence_ft as VAE_cubISft
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
from .jmvae_nf import JMVAE_NF_MNIST_SVHN as VAE_jnf_mnist_svhn
from .multi_vaes import Multi_VAES
from .mmvae import MMVAE_MNIST as VAE_mnist_fashion
from .jmvae_nf import JMVAE_NF_DCCA_MNIST_SVHN as VAE_jnf_mnist_svhn_dcca
from .jmvae_nf import JMVAE_NF_MNIST_CONTOUR as VAE_jnf_mnist_contour

__all__ = [ 'VAE_mnist_svhn',
            'VAE_circles_squares',
            'VAE_jnf_circles_squares',
            'VAE_jnf_mnist_fashion',
            'VAE_jnf_mnist_svhn',
            'VAE_mnist_fashion',
            'VAE_jnf_mnist_svhn_dcca',
            'VAE_jnf_mnist_contour'

            ]