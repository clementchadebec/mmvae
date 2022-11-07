# implement classes of joint_encoders, generic and specific to data for JMVAE, JMVAE_NF

import torch
from torch import nn
from bivae.models.multi_vaes import Multi_VAES
from bivae.utils import Constants
import torch.nn.functional as F
from .encoders import Encoder_VAE_MNIST
import numpy as np

def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))


class BaseEncoder(nn.Module):
    """ Simple MLP as a joint encoder """

    def __init__(self,size1, size2, hidden_dim, latent_dim, num_hidden_layers=1):
        super(BaseEncoder, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(size1 + size2, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        e = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        lv = self.fc22(e)
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta



class DoubleHeadMLP(nn.Module):
    """ Simple MLP with shared head as joint encoder"""

    def __init__(self, size1, size2, hidden_dim, latent_dim, num_hidden_layers=1):
        super(DoubleHeadMLP, self).__init__()
        self.input1 = nn.Sequential(nn.Linear(size1,hidden_dim), nn.ReLU(True))
        self.input2 = nn.Sequential(nn.Linear(size2,hidden_dim), nn.ReLU(True))
        modules = []
        modules.append(nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x0 = self.input1.forward(x[0].view(*x[0].size()[:-3], -1))
        x1 = self.input2.forward(x[1].view(*x[1].size()[:-3], -1))
        e = self.enc(torch.cat([x0,x1], dim=1)) # flatten data
        lv = torch.exp(0.5 * self.fc22(e))

        return self.fc21(e), lv + Constants.eta

class DoubleHeadJoint(nn.Module):

    def __init__(self, hidden_dim, params1, params2, encoder1, encoder2,args, state_dicts = [None, None]):
        super(DoubleHeadJoint, self).__init__()

        self.input1 = encoder1(params1)
        self.input2 = encoder2(params2)
        if state_dicts[0] is not None:
            print('Loading input1 network for the joint encoder')
            self.input1.load_state_dict(state_dicts[0])
        if state_dicts[1] is not None:
            print('Loading input2 network for the joint encoder')
            self.input2.load_state_dict(state_dicts[1])

        modules = []
        modules.append(nn.Sequential(nn.Linear(params1.latent_dim + params2.latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(args.num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, args.latent_dim)
        self.fc22 = nn.Linear(hidden_dim, args.latent_dim)

    def forward(self, x):
        x0 = self.input1.forward(x[0])[0]
        x1 = self.input2.forward(x[1])[0]
        e = self.enc(torch.cat([x0, x1], dim=1))  # flatten data
        lv = torch.exp(0.5 * self.fc22(e))
        return self.fc21(e), lv + Constants.eta
    
    

class MultipleHeadJoint(nn.Module):

    def __init__(self, hidden_dim, args_list, encoder_list,args, state_dicts = None):
        super(MultipleHeadJoint, self).__init__()

        self.inputs = nn.ModuleList([encoder_list[i](args_list[i]) for i in range(len(args_list))])
        
        assert(state_dicts is None) # No pretrained support yet

        modules = []
        joint_input_dim = np.sum([a.latent_dim for a in args_list])
        modules.append(nn.Sequential(nn.Linear(joint_input_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(args.num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, args.latent_dim)
        self.fc22 = nn.Linear(hidden_dim, args.latent_dim)

    def forward(self, x_list):

        xs = [self.inputs[i].forward(x)[0] for i, x in enumerate(x_list)]
        e = self.enc(torch.cat(xs, dim=1))  # flatten data
        lv = torch.exp(0.5 * self.fc22(e))
        return self.fc21(e), lv + Constants.eta

