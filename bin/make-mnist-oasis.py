# Link the indices of the two datasets to create a bimodal dataset

import torch
from torchvision import datasets, transforms
import pandas as pd

def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l2.unique():  # we take the labels from the OASIS dataset
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        print(n)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

if __name__ == '__main__':
    max_d = 300  # maximum number of datapoints per class
    dm = 7       # data multiplier: random permutations to match
    data_path = '/home/agathe/Code/datasets/'

    # get MNIST
    tx = transforms.ToTensor()
    train_mnist = datasets.MNIST(data_path, train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST(data_path, train=False, download=True, transform=tx)

    # get OASIS tsv files

    train_oasis = pd.read_csv(data_path + 'OASIS-1_dataset/tsv_files/lab_1/train_unbalanced.tsv', sep='\t')
    test_oasis = pd.read_csv(data_path + 'OASIS-1_dataset/tsv_files/lab_1/test_unbalanced.tsv', sep='\t')

    mnist_l, mnist_li = train_mnist.targets.sort()
    train_oasis['label'] = (train_oasis['diagnosis'] == 'AD')*1
    oasis_l, oasis_li = torch.tensor(train_oasis['label'].to_list()).sort()
    print('min max idx :', torch.min(oasis_li),torch.max(oasis_li) )
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, oasis_l, oasis_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2))
    torch.save(idx1, data_path+'mnist_oasis/train-mo-mnist-idx.pt')
    torch.save(idx2, data_path+'mnist_oasis/train-mo-oasis-idx.pt')

    # Repeat the operation on test dataset
    mnist_l, mnist_li = test_mnist.targets.sort()
    test_oasis['label'] = (test_oasis['diagnosis'] == 'AD') * 1
    oasis_l, oasis_li = torch.tensor(test_oasis['label'].to_list()).sort()

    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, oasis_l, oasis_li, max_d=max_d, dm=dm)
    shuffle = torch.randperm(len(idx1))
    print('len test idx:', len(idx1), len(idx2))
    torch.save(idx1[shuffle], data_path + 'mnist_oasis/test-mo-mnist-idx.pt')
    torch.save(idx2[shuffle], data_path + 'mnist_oasis/test-mo-oasis-idx.pt')
