import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms


from datasets import CUBImageFt


# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    fdir, fext = os.path.splitext(filepath)
    if hasattr(model, 'vaes'):
        for vae in model.vaes:
            save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)
            save_vars(vae.decoder.state_dict(), fdir + '_' + vae.modelName + '_decoder' + fext)
    if hasattr(model, 'joint_encoder'):
        save_vars(model.joint_encoder.state_dict(), fdir + '_joint_encoder' + fext)

def load_joint_vae(model, filepath):
    """ Load the state of joint autoencoders and decoders from previous training"""

    model.joint_encoder.load_state_dict(torch.load(filepath + 'model_joint_encoder.pt'))
    for i, vae in enumerate(model.vaes):
        vae.decoder.load_state_dict(torch.load(filepath + 'model_' + vae.modelName + '_decoder.pt' ))
    return


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def wasserstein_2(d1,d2):
    """ Computes the wasserstein distance between normal distributions"""
    if type(d1) != type(dist.Normal(0,1)) :
        raise NameError("Wasserstein_2 function must be applied to Normal distributions only.")

    w2 = (d1.mean - d2.mean)**2 + d1.stddev + d2.stddev - 2*torch.sqrt(d1.stddev*d2.stddev)

    return w2




def pdist(sample_1, sample_2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]


class FakeCategorical(dist.Distribution):
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, locs):
        self.logits = locs
        self._batch_shape = self.logits.shape

    @property
    def mean(self):
        return self.logits

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.logits.expand([*sample_shape, *self.logits.shape]).contiguous()

    def log_prob(self, value):
        # value of shape (K, B, D)
        lpx_z = -F.cross_entropy(input=self.logits.view(-1, self.logits.size(-1)),
                                 target=value.expand(self.logits.size()[:-1]).long().view(-1),
                                 reduction='none',
                                 ignore_index=0)

        return lpx_z.view(*self.logits.shape[:-1])
        # it is inevitable to have the word embedding dimension summed up in
        # cross-entropy loss ($\sum -gt_i \log(p_i)$ with most gt_i = 0, We adopt the
        # operationally equivalence here, which is summing up the sentence dimension
        # in objective.

def update_details(dict1, dict2):
    """Modify in place the first dict by adding values of the second dict"""
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k] += dict2[k]
        else :
            dict1[k] = dict2[k]

def update_dict_list(dict1, dict2):
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k].append(dict2[k])
        else :
            dict1[k] = [dict2[k]]

def get_mean_std(dict):
    return { k : torch.mean(torch.tensor(dict[k])) for k in dict.keys()}, {k : torch.std(torch.tensor(dict[k])) for k in dict.keys()}

def print_mean_std(dict_mean, dict_std):
    for k in dict_mean.keys():
        print(k, f' {dict_mean[k]} +- {dict_std[k]}')

def tensor_classes_labels(l1, l2, l1_names, l2_names):
    """ Transform labels that are tuples (l1[i], l2[i]) to int"""
    vl1, vl2 = len(np.unique(l1)), len(np.unique(l2))
    tlabels, tnames = [], []
    print(len(l1), len(l2))
    for i in range(len(l1)):
        tlabels.append(vl1*l2[i] + l1[i])
    for n2 in l2_names:
        for n1 in l1_names:
            tnames.append(n2 + n1)
    return tlabels, tnames


def extract_rayon(x: Tensor , eps = 1e-3):
    """x Tensor of size (bs x nb_data x ch x widht x height)
    return rayon tensor (nb_data x bs)"""
    rayons = torch.zeros(x.shape[0],x.shape[1])

    for i,batch in enumerate(x):
        for j,d in enumerate(batch):
            d = (d.squeeze() >= eps).int()
            r = torch.argmax(d[15,:])
            rayons[i,j] = (32-2*r)/(32)
    return rayons.permute(1,0)

def check_parameters_equal(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print("the models have different parameters")
            return
    print("the models have equal parameters")

def check_non_zero_parameters(m):
    for p in m.parameters() :
        if p.grad.data.ne(torch.zeros_like(p.grad.data)).sum() > 0:
            print("the model have non-zero grad")
            return
    print("the model has zero grad")


def negative_entropy(rayons, range, bins):
    """Compute an approximate entropy from the samples to compare the sizes of the samples to the
    uniform distribution

    rayons is array of shape n_data, n_samples"""

    entropy = 0
    for data in rayons:
        p = np.histogram(data, range=range, bins=bins, density=False)[0] + 1e-5
        p/=len(data)
        entropy += np.sum(np.log(p)*p)
    return entropy/len(rayons)


class add_channels(object):

    def __call__(self, image):
        if image.shape[0] == 1:
            image = torch.cat([image, torch.zeros_like(image), torch.zeros_like(image)], dim=0)
        return image



def adjust_shape(data_1, data_2):

    # first adjust the number of channels
    if data_1.shape[1] > data_2.shape[1]:
        data_2 = torch.cat([data_2 for _ in range(data_1.shape[1])], dim=1)
    elif data_2.shape[1] > data_1.shape[1]:
        data_1 = torch.cat([data_1 for _ in range(data_2.shape[1])], dim=1)

    # Then adjust w and h
    w1,w2 = data_1.shape[2], data_2.shape[2]
    h1,h2 = data_1.shape[3], data_2.shape[3]
    h,w = max(h1,h2), max(w1,w2)
    data_1 = F.pad(data_1, ((h-h1)//2, (h-h1)//2, (w-w1)//2, (w-w1)//2), mode='constant',value=0)
    data_2 = F.pad(data_2, ((h-h2)//2, (h-h2)//2, (w-w2)//2, (w-w2)//2), 'constant', 0)
    return data_1, data_2

