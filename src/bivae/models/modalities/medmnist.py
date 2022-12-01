
import torch
import numpy as np


from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.utils import unpack_data, add_channels
from torchvision import transforms
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader



class medmnist_utils():
    
    
    def __init__(self) -> None:
        
        self.shape_mods = [(3,28,28), (1,28,28)]
        
        
        pass
    
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        len_train = None
        if hasattr(self.params, 'len_train'):
            len_train = self.params.len_train
        train, test, val = MNIST_SVHN_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform, len_train=len_train)
        return train, test, val
    
    def set_classifiers(self):

        self.classifiers = [load_pretrained_mnist(), load_pretrained_svhn()]
        return 
    
    def compute_fid(self, batch_size):
        return 