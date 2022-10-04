import torch
from torch import nn


class wrapper_pythae_model(nn.Module):

    def __init__(self, model):
        super(wrapper_pythae_model, self).__init__()
        self.model = model

    def forward(self, x):
        self.model.eval()
        pred = self.model.encoder(x)[0] # if VAE encoder #batchsize x embedding_size
        return pred.cpu().numpy()

