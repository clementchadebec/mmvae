from pathlib import Path
import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
import datetime
from bivae.utils import accuracy
import argparse
from bivae.models.nn.medmnist import Encoder_ResNet_AE_medmnist
from pythae.models.base import BaseAEConfig
from torch.nn import functional as F
from medmnist import PathMNIST, BloodMNIST
from torchvision.models import resnet18, ResNet18_Weights
from bivae.utils import add_channels
from tqdm import tqdm
from bivae.models.nn.medmnist_classifiers import ResNet18

'''
Load and wrap the classifiers for PATHMnist, BloodMnist.
'''





class ClassifierPATH(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        print('loading weights')
        weights = torch.load('../medmnist_models/resnet18_28_1_path.pth')
        self.network = ResNet18(3,9)
        self.network.load_state_dict(weights['net'])
        self.transform = transforms.Normalize(mean=[.5], std=[.5])
        
    def forward(self, x):
        return self.network(self.transform(x))
    

class ClassifierBLOOD(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        print('loading weights')
        weights = torch.load('../medmnist_models/resnet18_28_1_blood.pth')
        self.network = ResNet18(3,8)
        self.network.load_state_dict(weights['net'])
        self.transform = transforms.Normalize(mean=[.5], std=[.5])
        
    def forward(self, x):
        return self.network(self.transform(x))




##############################################################################################################
# Load the fashion mnist dataset and train



if __name__ == '__main__':
    
    

    # tx = transforms.Compose([transforms.ToTensor(), add_channels()])  # Mnist is already with values between 0 and 1
    
    test_set_p = PathMNIST('test', transforms.ToTensor())
    print(len(test_set_p))
    model_p = ClassifierPATH()
    test_loader_p = DataLoader(test_set_p,
                             batch_size=2000, shuffle=False)
    
    test_set_b = BloodMNIST('test', transforms.ToTensor())
    print(len(test_set_b))
    model_b = ClassifierBLOOD()
    test_loader_b = DataLoader(test_set_b,
                             batch_size=2000, shuffle=False)


    def test(test_loader, model):
        model.eval()
        loss, acc = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):

                data, labels = data[0], data[1].squeeze()
                logits = model(data)
                acc += accuracy(logits, labels)

        print(
            f"====>  Accuracy on the test set {acc / len(test_loader.dataset)}")
        return 

    


    test(test_loader_p, model_p)
    test(test_loader_b, model_b)


