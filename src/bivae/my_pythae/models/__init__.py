""" 
This is the heart of pythae! 
Here are implemented some of the most common (Variational) Autoencoders models.

By convention, each implemented model is stored in a folder located in :class:`pythae.models`
and named likewise the model. The following modules can be found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from either :class:`~pythae.models.base.AEConfig` for Autoencoder models or 
    :class:`~pythae.models.base.VAEConfig` for Variational Autoencoder models. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.AE` for Autoencoder models or 
    :class:`~pythae.models.base.VAE` for Variational Autoencoder models. 
- *modelname_utils.py* (optional): A module where utils methods are stored.
"""


from .vae import VAE, VAEConfig
from .base import BaseAE, BaseAEConfig

from .vae import VAE, VAEConfig, my_VAE
from .vae_iaf import VAE_IAF, VAE_IAF_Config, my_VAE_IAF
from .vae_lin_nf import VAE_LinNF, VAE_LinNF_Config, my_VAE_LinNF


__all__ = [
    "BaseAE",
    "BaseAEConfig",

    "VAE",
    "VAEConfig",

    "my_VAE_LinNF",
    "VAE_LinNF",
    "VAE_LinNF_Config",
    "VAE_IAF",
    "VAE_IAF_Config",
    "my_VAE_IAF",
    "my_VAE"
]
