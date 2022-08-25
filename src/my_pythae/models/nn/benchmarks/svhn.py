"""Proposed neural nets architectures suited for MNIST"""

from typing import List

import torch
import torch.nn as nn

from ....models.base.base_utils import ModelOutput
from ..base_architectures import BaseDecoder, BaseEncoder



class Encoder_VAE_SVHN(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = (3,32,32)
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]
        self.fBase = 32

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(self.n_channels, self.fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(self.fBase, self.fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(self.fBase * 2, self.fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4

        )
        self.c1 = nn.Conv2d(self.fBase*4, self.latent_dim, 4, 2,0)
        self.c2 = nn.Conv2d(self.fBase*4, self.latent_dim, 4, 2,0)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x :  torch.Tensor):
        e = self.enc(x)
        mu = self.c1(e).reshape(-1, self.latent_dim)
        lv = self.c2(e).reshape(-1, self.latent_dim)
        output = ModelOutput(
            embedding=mu,
            log_covariance=lv
        )
        return output