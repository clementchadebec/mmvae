# Make a toy multimodal dataset where one modality is mnist and the other is fashion mnist

import torch
from torchvision import datasets, transforms
import numpy as np
import json

unbalanced = False
# in place i the labels of fashion mnist that are associated with the number i in mnist

correspondence = [[1,2,3], [4,5,6], [7,8,9]] if unbalanced else np.arange(10).reshape(-1,1)
output_path = '../data/unbalanced/' if unbalanced else '../data/'

def values_in(arr, filter):
    """ Returns a boolean array that indicates which values of array are in filter"""

    bool_arr = torch.zeros_like(arr)
    for v in filter:
        bool_arr += arr == v
    return bool_arr.bool()


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    dm = number of time to repeat the randomization process
    """
    _idx1, _idx2 = [], []
    for l in range(len(correspondence)): # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[values_in(l2,correspondence[l])]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[torch.randperm(l_idx1.size(0))][:n], l_idx2[torch.randperm(l_idx2.size(0))][:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

if __name__ == '__main__':
    max_d = 5000  # maximum number of datapoints per class
    dm = 30        # data multiplier: random permutations to match

    # get the individual datasets
    tx = transforms.ToTensor()
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST('../data', train=False, download=True, transform=tx)
    train_fashion = datasets.FashionMNIST('../data', train=True, download=True, transform=tx)
    test_fashion = datasets.FashionMNIST('../data', train=False, download=True, transform=tx)

    mnist_l, mnist_li = train_mnist.targets.sort()
    fashion_l, fashion_li = train_fashion.targets.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, fashion_l, fashion_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2))
    torch.save(idx1, output_path + 'train-ms-mnist-idx.pt')
    torch.save(idx2, output_path + 'train-ms-fashion-idx.pt')

    mnist_l, mnist_li = test_mnist.targets.sort()
    fashion_l, fashion_li = test_fashion.targets.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, fashion_l, fashion_li, max_d=max_d, dm=dm)
    print('len test idx:', len(idx1), len(idx2))
    torch.save(idx1, output_path + 'test-ms-mnist-idx.pt')
    torch.save(idx2, output_path + 'test-ms-fashion-idx.pt')


    print(idx1[:20], idx2[:20])

