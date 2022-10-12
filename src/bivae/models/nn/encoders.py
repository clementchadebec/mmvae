# Define custom encoders architectures for the VAEs
import torch
from torch import nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class Encoder_VAE_MNIST(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        self.fBase = 32


        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, self.fBase, 4, 2, padding=1),
            # fBase x 14 x 14
            nn.BatchNorm2d(self.fBase),
            nn.ReLU(),
            nn.Conv2d(self.fBase, 2*self.fBase, 4, 2, padding=1),
            # 64 x 7 x 7
            nn.BatchNorm2d(2*self.fBase),
            nn.ReLU(),
            nn.Conv2d(2*self.fBase, 4*self.fBase, 5, 2, padding=1),
            # 128 x 3 x 3
            nn.BatchNorm2d(4*self.fBase),
            nn.ReLU())

        self.embedding = nn.Linear(4*self.fBase * 3*3, args.latent_dim)
        self.log_var = nn.Linear(4*self.fBase * 3*3, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_AE_MNIST(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        self.fBase = 32

        self.fc = nn.Linear(args.latent_dim, self.fBase*4 * 4 * 4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(4*self.fBase, 2*self.fBase, 3, 2, padding=1),
            nn.BatchNorm2d(2*self.fBase),
            nn.ReLU(),
            nn.ConvTranspose2d(2*self.fBase, self.fBase, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.fBase),
            nn.ReLU(),
            nn.ConvTranspose2d(self.fBase, self.n_channels, 3, 2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], self.fBase*4, 4, 4)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output



class Encoder_VAE_SVHN(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
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


class Decoder_VAE_SVHN(BaseDecoder):

    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.fBase = 32
        self.nb_channels = args.input_dim[0]

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(self.fBase * 4, self.fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(self.fBase * 2, self.fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(self.fBase, self.nb_channels, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z : torch.Tensor):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        output = ModelOutput(reconstruction = out)
        return output



class TwoStepsDecoder(BaseDecoder):

    def __init__(self,decoder, config, state_dict, args):
        BaseDecoder.__init__(self)

        self.pretrained_decoder = decoder(config)
        if state_dict is not None :
            self.pretrained_decoder.load_state_dict(state_dict)
        self.nb_hidden = args.num_hidden_layers
        self.hidden_dim = 512
        self.latent_dim = args.latent_dim

        self.modules = [
            nn.Linear(self.latent_dim,self.hidden_dim),
            nn.ReLU()
        ]
        for i in range(self.nb_hidden - 1):
            self.modules.extend([
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.ReLU()
            ])
        self.modules.extend([nn.Linear(self.hidden_dim, self.pretrained_decoder.latent_dim), nn.ReLU()])
        self.first_step_decoder = nn.Sequential(
            *self.modules
        )

        # Fix grads for the second_step_decoder
        # self.pretrained_decoder.requires_grad_(False)

    def forward(self,z):

        z1 = self.first_step_decoder(z)
        x = self.pretrained_decoder(z1)
        return x


class TwoStepsEncoder(BaseEncoder):
    """Defines a two-step encoder, with the first step being pretrained and requires no grad"""
    def __init__(self, pretrained_encoder,args):

        BaseEncoder.__init__(self)
        self.first_encoder = pretrained_encoder
        self.hidden_dim = 512
        self.num_hidden = 3
        self.latent_dim = args.latent_dim
        self.modules = [
            nn.Linear(self.first_encoder.latent_dim, self.hidden_dim),
            nn.ReLU()
        ]
        for i in range(self.num_hidden - 1):
            self.modules.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ])
        self.sec_encoder = nn.Sequential(*self.modules)
        self.embedding = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.first_encoder.requires_grad_(False)

    def forward(self,x):

        with torch.no_grad():
            h = self.first_encoder(x).embedding
        h1 = self.sec_encoder(h)
        out = ModelOutput(
            embedding = self.embedding(h1),
            log_covariance = self.log_var(h1)
        )
        return out


########################################################################################################################
################################################## OASIS DATASET #######################################################


class encoder_OASIS(nn.Module):

    def __init__(self, args):
        super(encoder_OASIS, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.input_dim,1000), nn.ReLU(),
            nn.Linear(1000,400), nn.ReLU(),
        )
        self.mu = nn.Linear(400,args.latent_dim,bias=True)
        self.lcov = nn.Linear(400,args.latent_dim, bias=True)

    def forward(self, x: torch.Tensor):
        h1 = self.layers(x)
        output = ModelOutput(
            embedding=self.mu(h1),
            log_covariance=self.lcov(h1)
        )
        return output


class decoder_OASIS(nn.Module):

    # To define
    def __init__(self, args):
        super(decoder_OASIS, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim,400), nn.ReLU(),
            nn.Linear(400,1000), nn.ReLU(),
            nn.Linear(1000, args.input_dim), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):
        output = ModelOutput(reconstruction=self.layers(z))
        return output


