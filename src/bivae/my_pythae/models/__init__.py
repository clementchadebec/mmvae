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



from .vae import my_VAE
from .vae_iaf import my_VAE_IAF
from .vae_lin_nf import my_VAE_LinNF
from .laplace_vae import laplace_VAE
from .vae_maf import my_VAE_MAF, VAE_MAF_Config

__all__ = [

    "my_VAE_LinNF",
    "my_VAE_IAF",
    "my_VAE",
    "laplace_VAE", 
    "my_VAE_MAF", 
    "VAE_MAF_Config"
]
