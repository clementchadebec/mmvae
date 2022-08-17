# Dataloaders classes to be used with any model

from torch.utils.data import DataLoader
import os
import torch
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import datasets, transforms
from datasets import CIRCLES_DATASET
from torch.utils.data import random_split


##########################################################################################################
#################################### UNIMODAL DATALOADERS ################################################


class MNIST_DL():
    def __init__(self, data_path, type):
        self.type = type
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True,device='cuda', transform=None):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == "cuda" else {}
        if transform is None:
            tx = transforms.ToTensor()
        else :
            tx = transform
        datasetC = datasets.MNIST if self.type == 'numbers' else datasets.FashionMNIST
        train = DataLoader(datasetC('../data', train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC('../data', train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

class BasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform = None):

        self.data = data # shape len_data x ch x w x h
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0 # We return 0 as label to have a dataset that is homogeneous with the other datasets

class SVHN_DL():

    def __init__(self, data_path = '../data'):
        self.data_path = data_path
        return

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

        train = DataLoader(datasets.SVHN('/home/agathe/Code/Datasets/svhn', split='train', download=True, transform=transform),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN('/home/agathe/Code/Datasets/svhn', split='test', download=True, transform=transform),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

class CIRCLES_DL():

    def __init__(self, type, data_path):
        self.type = type
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        # create datasets
        train_set = CIRCLES_DATASET(self.data_path + self.type + '_train.pt', self.data_path + 'labels_train.pt',
                                    self.data_path + 'r_' + self.type + '_train.pt',
                                    transforms=transform)
        test_set = CIRCLES_DATASET(self.data_path + self.type + '_test.pt', self.data_path + 'labels_test.pt',
                                   self.data_path + 'r_' + self.type + '_test.pt',
                                   transforms=transform)
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test


########################################################################################################################
####################################### MULTIMODAL DATALOADERS #########################################################

class MultimodalBasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        # data of shape n_mods x len_data x ch x w x h
        self.lenght = len(data[0])
        self.datasets = [BasicDataset(d, transform) for d in data ]

    def __len__(self):
        return self.lenght

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)

class CIRCLES_SQUARES_DL():

    def __init__(self, data_path):
        self.data_path=data_path

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform=None):
        # load base datasets
        t1, s1 = CIRCLES_DL('squares', self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = CIRCLES_DL('circles', self.data_path).getDataLoaders(batch_size, shuffle, device, transform)

        train_circles_discs = TensorDataset([t1.dataset, t2.dataset])
        test_circles_discs = TensorDataset([s1.dataset, s2.dataset])

        # Split the test and val with always the same seed for reproducibility
        val_set, test_set = random_split(test_circles_discs,
                                         [len(test_circles_discs) // 2,
                                          len(test_circles_discs) - len(test_circles_discs) // 2],
                                         generator=torch.Generator().manual_seed(42))

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_circles_discs, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val

class MNIST_FASHION_DATALOADER():

    def __init__(self, data_path):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=None):
        print(self.data_path)
        if not (os.path.exists(self.data_path + 'train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'train-ms-fashion-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + 'test-ms-fashion-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + 'train-ms-mnist-idx.pt')
        t_fashion = torch.load(self.data_path + 'train-ms-fashion-idx.pt')
        s_mnist = torch.load(self.data_path + 'test-ms-mnist-idx.pt')
        s_fashion = torch.load(self.data_path + 'test-ms-fashion-idx.pt')

        # load base datasets
        t1,s1 = MNIST_DL(self.data_path,'numbers').getDataLoaders(batch_size,shuffle,device, transform)
        t2,s2 = MNIST_DL(self.data_path,'fashion').getDataLoaders(batch_size,shuffle,device, transform)


        train_mnist_fashion = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_fashion[i], size=len(t_fashion))
        ])

        test_mnist_fashion = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_fashion[i], size=len(s_fashion))
        ])

        val_set, test_set = random_split(test_mnist_fashion,
                                         [len(test_mnist_fashion)//2,
                                          len(test_mnist_fashion)-len(test_mnist_fashion)//2],
                                         generator=torch.Generator().manual_seed(42)
                                         )

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_fashion, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val

class MNIST_SVHN_DL():

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):

        if not (os.path.exists(self.data_path + '/train-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/train-ms-svhn-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-mnist-idx.pt')
                and os.path.exists(self.data_path + '/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + '/train-ms-mnist-idx.pt')
        t_svhn = torch.load(self.data_path + '/train-ms-svhn-idx.pt')
        s_mnist = torch.load(self.data_path + '/test-ms-mnist-idx.pt')
        s_svhn = torch.load(self.data_path + '/test-ms-svhn-idx.pt')

        # load base datasets
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)

        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        # Split between test and validation while fixing the seed to ensure that we always have the same sets
        val_set, test_set = random_split(test_mnist_svhn,
                                         [len(test_mnist_svhn) // 2,
                                          len(test_mnist_svhn) - len(test_mnist_svhn) // 2],
                                         generator=torch.Generator().manual_seed(42))



        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val