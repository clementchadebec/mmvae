"""
In this module are stored the main Neural Networks Architectures.
"""


from .base_architectures import (BaseDecoder, BaseDiscriminator, BaseEncoder,
                                 BaseMetric)

from .default_architectures import (Encoder_VAE_MLP,Decoder_AE_MLP)

from .benchmarks import Encoder_VAE_SVHN

__all__ = ["BaseDecoder", "BaseEncoder", "BaseMetric", "BaseDiscriminator",
           "Encoder_VAE_MLP", "Decoder_AE_MLP",
           "Encoder_VAE_SVHN"]
