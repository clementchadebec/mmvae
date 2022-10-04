from .classifier_mnist import MnistClassifier
from .classifier_SVHN import SVHNClassifier
from .classifier_empty_full import CirclesClassifier
from .Quality_assess import GenerativeQualityAssesser, Inception_quality_assess, custom_mnist_fashion
from .accuracies import compute_accuracies

__all__ = [
    'MnistClassifier',
    'SVHNClassifier',
    'CirclesClassifier',
    'GenerativeQualityAssesser',
    'Inception_quality_assess',
    'custom_mnist_fashion'
    'compute_accuracies'


]