# Dataloaders classes to be used with any model

from torch.utils.data import DataLoader, Dataset
import os
import torch
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import pandas as pd
from bivae.data_utils.transforms import contour_transform, random_grey_transform, binary_transform
import numpy as np
from medmnist import PathMNIST, TissueMNIST

########################################################################################################################
########################################## DATASETS ####################################################################


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

class MultimodalBasicDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):
        # data of shape n_mods x len_data x ch x w x h
        self.lenght = len(data[0])
        self.datasets = [BasicDataset(d, transform) for d in data ]

    def __len__(self):
        return self.lenght

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)


# Dataset for circles and discs

class CIRCLES_DATASET(Dataset):

    def __init__(self, data_path, labels_path,r_path, transforms=None):
        super().__init__()
        self.data = torch.load(data_path) # tensor of size nb_sample, size_image, size_image
        self.labels = torch.load(labels_path)
        self.r_path = torch.load(r_path)
        self.transforms = transforms
    def __getitem__(self, item):

        sample = [self.data[item], self.labels[item], self.r_path[item]]
        if self.transforms is not None:
            sample[0] = self.transforms(sample[0])
        return tuple(sample)

    def __len__(self):
        return len(self.data)


class MRIDataset(Dataset):

    def __init__(self, img_dir, data_df, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data_df = data_df
        self.label_code = {"AD": 1, "CN": 0}

        self.size = self[0]['image'].shape

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        diagnosis = self.data_df.loc[idx, 'diagnosis']
        label = self.label_code[diagnosis]

        participant_id = self.data_df.loc[idx, 'participant_id']
        session_id = self.data_df.loc[idx, 'session_id']
        filename = participant_id + '_' + session_id + \
                   '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt'

        image = torch.load(os.path.join(self.img_dir, filename))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant_id,
                  'session_id': session_id}
        return sample

    def train(self):
        self.transform.train()

    def eval(self):
        self.transform.eval()

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
        train = DataLoader(datasetC(self.data_path , train=True, download=True, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasetC(self.data_path, train=False, download=True, transform=tx),
                          batch_size=batch_size, shuffle=False, **kwargs)
        return train, test

class SVHN_DL():

    def __init__(self, data_path = '../data'):
        self.data_path = data_path
        return

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

        train = DataLoader(datasets.SVHN(self.data_path, split='train', download=True, transform=transform),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN(self.data_path , split='test', download=True, transform=transform),
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

class MNIST_FASHION_DL():

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

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor(), len_train=None):

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
        
        # shuffle to be able to reduce size of the dataset
        rd_idx = np.random.permutation(len(t_mnist))
        t_mnist, t_svhn = t_mnist[rd_idx], t_svhn[rd_idx]
        if len_train is None: 
            len_train = len(t_mnist)
        
        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len_train),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len_train)
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        # Split between test and validation while fixing the seed to ensure that we always have the same sets
        len_val = min(10000, len(train_mnist_svhn)//10)
        train_set, val_set = random_split(train_mnist_svhn,
                                         [len(train_mnist_svhn)-len_val,
                                          len_val],
                                         generator=torch.Generator().manual_seed(42))



        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val

class BINARY_MNIST_SVHN_DL():

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
        transf_mnist = transforms.Compose([transforms.ToTensor(),binary_transform() ])
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transf_mnist)
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

class MNIST_OASIS_DL():

    def __init__(self, data, oasis_transform = None, mnist_transform = None):
        if data not in ['balanced', 'unbalanced']:
            raise ValueError('Data can either be "balanced" or "unbalanced"')
        self.name = 'mnist_oasis_dl'
        self.oasis_transform = oasis_transform
        self.mnist_transform = mnist_transform
        self.data_type = data

    def getDataLoaders(self,batch_size, shuffle=True, device='cuda'):

        # get the linked indices
        t_mnist = torch.load('mnist_oasis/data/' + self.data_type +'/train-mo-mnist-idx.pt')
        s_mnist = torch.load('mnist_oasis/data/'+ self.data_type + '/test-mo-mnist-idx.pt')
        t_oasis = torch.load('mnist_oasis/data/' + self.data_type +'/train-mo-oasis-idx.pt')
        s_oasis = torch.load('mnist_oasis/data/' +self.data_type + '/test-mo-oasis-idx.pt')


        # Get the base datasets
        t1,s1 = MNIST_DL('home/Code/vaes/mmvae/data', type='numbers').getDataLoaders(batch_size,shuffle,
                                                                                     device, self.mnist_transform)
        oasis_path = '/home/agathe/Code/datasets/OASIS-1_dataset/'
        train_df = pd.read_csv(oasis_path+'tsv_files/lab_1/train_{}.tsv'.format(self.data_type), sep='\t')
        test_df = pd.read_csv(oasis_path + 'tsv_files/lab_1/test_{}.tsv'.format(self.data_type), sep ='\t')



        t2 = MRIDataset(oasis_path + 'preprocessed',train_df,transform=self.oasis_transform)
        s2 = MRIDataset(oasis_path + 'preprocessed', test_df, self.oasis_transform)

        # Create the paired dataset

        train_mnist_oasis = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2,lambda d,i : t_oasis[i], size=len(t_oasis))
            # t2
        ])

        test_mnist_oasis = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2, lambda d, i: s_oasis[i], size=len(s_oasis))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_oasis, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_oasis, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test


class MNIST_CONTOUR_DL():

    def __init__(self, data_path='/home/agathe/Code/datasets'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):


        # load base datasets

        # Simple MNIST dataset
        t1,s1 = MNIST_DL(self.data_path,'numbers').getDataLoaders(batch_size,shuffle,device, random_grey_transform)
        # Dataset with Canny Transform
        t2,s2 = MNIST_DL(self.data_path,'numbers').getDataLoaders(batch_size,shuffle,device, contour_transform)


        train_mnist_fashion = TensorDataset([
            t1.dataset,t2.dataset
        ])

        test_mnist_fashion = TensorDataset([
            s1.dataset,s2.dataset
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


class CELEBA_MASK_DL():

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        # to fill
        return


from .datasets import CelebA
from torchvision.transforms import ToTensor
class CELEBA_DL():

    def __init__(self, data_path='../data/'):
        self.data_path = data_path 
    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', len_train=None, transform=ToTensor()):

        train_dataset = CelebA(self.data_path, 'train', transform=transform, len=len_train)
        test = CelebA(self.data_path, 'test', transform=transform)
        val = CelebA(self.data_path, 'val', transform=transform)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle, **kwargs)
        test_dataloader = DataLoader(test, batch_size, shuffle=False, **kwargs)
        val_dataloader = DataLoader(val, batch_size,shuffle=False,**kwargs )
        return train_dataloader, test_dataloader, val_dataloader



class MNIST_SVHN_FASHION_DL():

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):

        if not (os.path.exists(self.data_path + '/train-msf-mnist-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load(self.data_path + '/train-msf-mnist-idx.pt')
        t_svhn = torch.load(self.data_path + '/train-msf-svhn-idx.pt')
        t_fashion = torch.load(self.data_path + '/train-msf-fashion-idx.pt')
        s_mnist = torch.load(self.data_path + '/test-msf-mnist-idx.pt')
        s_svhn = torch.load(self.data_path + '/test-msf-svhn-idx.pt')
        s_fashion = torch.load(self.data_path + '/test-msf-fashion-idx.pt')

        # load base datasets
        t1, s1 = MNIST_DL(self.data_path, type='numbers').getDataLoaders(batch_size, shuffle, device, transform)
        t2, s2 = SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        t3, s3 = MNIST_DL(self.data_path, type='fashion').getDataLoaders(batch_size, shuffle,device, transform)

        # shuffle to be able to reduce size of the dataset
        rd_idx = np.random.permutation(len(t_mnist))
        t_mnist, t_svhn, t_fashion = t_mnist[rd_idx], t_svhn[rd_idx], t_fashion[rd_idx]
        
        rd_idx_test = np.random.permutation(len(s_mnist))
        s_mnist, s_svhn, s_fashion = s_mnist[rd_idx_test], s_svhn[rd_idx_test], s_fashion[rd_idx_test]
        
        # len_train = 100000
        len_train = len(t_mnist)
        
        # len_test = 1000
        len_test = len(s_mnist)
        
        train_msf = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len_train),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len_train),
            ResampleDataset(t3.dataset, lambda d,i : t_fashion[i], size=len_train)
        ])
        
        test_msf = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len_test),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len_test),
            ResampleDataset(s3.dataset, lambda d,i : s_fashion[i], size=len_test)
        ])

        # Split between test and validation while fixing the seed to ensure that we always have the same sets
        train_set, val_set = random_split(train_msf,
                                         [len(train_msf)-10000,
                                          10000],
                                         generator=torch.Generator().manual_seed(42))



        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_msf, batch_size=batch_size, shuffle=False, **kwargs)
        val = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
        return train, test, val



class PATH_TISSUE_DL():
    
    def __init__(self) -> None:
        pass
    
    # def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
    #     dl = {}
    #     id1, id2 = {},{}
    #     # get the training datasets
    #     for j,split in enumerate(['train', 'test', 'val']):
    #         d1 = PathMNIST(split=split, transform=transform)
    #         d2 = TissueMNIST(split=split, transform=transform)
            
    #         # Get the indices
    #         id1.append(torch.load('../data/{}-med-path-idx.pt'.format(split)))
    #         id2.append(torch.load('../data/{}-med-tissue-idx.pt'.format(split)))

    #     # Resample
    #     dl['train'] = TensorDataset([
    #         ResampleDataset(d1, lambda d,i : id1[j][i], size=len(id1[j])), 
    #         ResampleDataset(d2, lambda d,i : id2[j][i], size=len(id2[j]))
    #     ])

    #     kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    #     train = DataLoader(dl['train'], batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     test = DataLoader(dl['test'], batch_size=batch_size, shuffle=False, **kwargs)
    #     val = DataLoader(dl['val'], batch_size=batch_size, shuffle=False, **kwargs)
    #     return train, test, val
    
    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        
        d1_train = PathMNIST('train',transform=transform)
        d2_train = TissueMNIST('train', transform=transform)
        
        id1_train = torch.load('../data/train-med-path-idx.pt')
        id2_train = torch.load('../data/train-med-tissue-idx.pt')
        
        tensor_train = TensorDataset([
            ResampleDataset(d1_train, lambda d,i : id1_train[i], size=len(id1_train)), 
            ResampleDataset(d2_train, lambda d, i : id2_train[i], size=len(id2_train))
        ])
        
        train_dl = DataLoader(tensor_train, batch_size=batch_size, shuffle=True)
        
        d1_test = PathMNIST('test',transform=transform)
        d2_test = TissueMNIST('test', transform=transform)
        
        id1_test = torch.load('../data/test-med-path-idx.pt')
        id2_test = torch.load('../data/test-med-tissue-idx.pt')
        
        tensor_test = TensorDataset([
            ResampleDataset(d1_test, lambda d,i : id1_test[i], size=len(id1_test)), 
            ResampleDataset(d2_test, lambda d, i : id2_test[i], size=len(id2_test))
        ])
        
        test_dl = DataLoader(tensor_test, batch_size=batch_size, shuffle=False)
        
        d1_val = PathMNIST('val',transform=transform)
        d2_val = TissueMNIST('val', transform=transform)
        
        id1_val = torch.load('../data/val-med-path-idx.pt')
        id2_val = torch.load('../data/val-med-tissue-idx.pt')
        
        tensor_val = TensorDataset([
            ResampleDataset(d1_train, lambda d,i : id1_val[i], size=len(id1_val)), 
            ResampleDataset(d2_train, lambda d, i : id2_val[i], size=len(id1_val))
        ])
        
        val_dl = DataLoader(tensor_val, batch_size=batch_size, shuffle=False)
    
        return train_dl, test_dl, val_dl