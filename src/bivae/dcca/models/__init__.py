from .mnist_svhn import DeepCCA_MNIST_SVHN, load_dcca_mnist_svhn
from .mnist_contour import DeepCCA_MNIST_CONTOUR, load_dcca_mnist_contour
from .celeba import DeepCCA_celeba, load_dcca_celeba
from .medmnist import DeepCCA_MedMNIST, load_dcca_medmnist

__all__ = [
    'DeepCCA_MNIST_SVHN',
    'load_dcca_mnist_svhn',
    'DeepCCA_MNIST_CONTOUR',
    'load_dcca_mnist_contour',
    'DeepCCA_celeba',
    'load_dcca_celeba', 
    'DeepCCA_MedMNIST', 
    'load_dcca_medmnist'
]