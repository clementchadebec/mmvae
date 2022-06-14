"""
Use Frechet Inception Score to evaluate the generative model.
Adapt the methodology for multimodal evaluation.

"""

from torch import nn
import torch
from analysis.pytorch_fid.inception import InceptionV3



class WrapperDoubleInception3(nn.Module):

    def __init__(self, dims, device='cuda'):
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.fid_incv3 = InceptionV3([block_idx]).to(device)
        self.dims = dims

    @torch.no_grad()
    def forward(self, data):

        f = []
        for x in data:
            y = self.fid_incv3(x)
            y = y[0]
            y = y[:,:,0,0]
            print(y.shape)
            f.append(y)
        return torch.cat(f, dim=1)

class mFID():

    def __init__(self, features_extractor):
        self.features_extractor = features_extractor


    def compute_score(self, data1, data2):

        f1 = self.features_extractor(data1)
        f2 = self.features_extractor(data2)
        print(f1.shape)
        return
