from pydantic.dataclasses import dataclass
from torch import nn
from ..my_pythae.models.vae import VAEConfig, my_VAE
from pythae.models.base import base_model

@dataclass
class JMVAE_NF_CONFIG :

    name : str = None
    joint_encoder : nn.Module = None
    params : object = None
    vae : base_model = my_VAE

    # First VAE specification
    encoder1 : nn.Module = None
    decoder1 : nn.Module = None
    vae_config1 : VAEConfig

    # Second VAE specification
    encoder2 : nn.Module = None
    decoder2 : nn.Module = None
    vae_config2 : VAEConfig

    vaes = nn.ModuleList([
        vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
        vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

    ])


test_config = JMVAE_NF_CONFIG(name = 'test_config')

print(test_config)