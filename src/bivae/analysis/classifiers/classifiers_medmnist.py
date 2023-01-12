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
from medmnist import PneumoniaMNIST, BloodMNIST
from torchvision.models import resnet18, ResNet18_Weights
from bivae.utils import add_channels
from tqdm import tqdm
from bivae.models.nn.medmnist_classifiers import ResNet18
import numpy as np
from torchnet.dataset import TensorDataset, ResampleDataset
from pythae.models.base.base_utils import ModelOutput




'''
Load and wrap the classifiers for PATHMnist, BloodMnist.
'''





class ClassifierPneumonia(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        print('loading weights')
        weights = torch.load('../medmnist_models/resnet18_28_2_pneumonia.pth')
        self.network = ResNet18(3,2)
        self.network.load_state_dict(weights['net'])

        self.transform = transforms.Compose([ 
        transforms.Normalize(
            mean=[.5], std=[.5])
        ,transforms.Lambda(lambda x: x.repeat(1,3,1,1))
        ])
        
    def forward(self, x):
        #print(x.shape)
        
        h = self.network(self.transform(x))
        #print(h.shape)
        return h
    

class ClassifierBLOOD(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        print('loading weights')
        weights = torch.load('../medmnist_models/resnet18_28_1_blood.pth')
        self.network = ResNet18(3,8)
        self.network.load_state_dict(weights['net'])
        self.network.linear = nn.Linear(512,2)
        self.transform = transforms.Normalize(mean=[.5], std=[.5])
        
    def forward(self, x):
        return self.network(self.transform(x))

2
def load_medmnist_classifiers():
    
    model1 = ClassifierPneumonia()
    model2 = ClassifierBLOOD()
    model2.load_state_dict(torch.load('../experiments/classifiers_medmnist/model_2.pt'))
    
    model1.eval()
    model2.eval()

    model1.cuda()
    model2.cuda()
    
    return [model1, model2]


#############################################################################################################
# Define a fake DCCA with trained classifiers to wether the problems comes from the quality of the 
# information extracted with the DCCA or is more fundamental than that

class fake_encoder_lcca_model1(nn.Module):

    def __init__(self):
        super(fake_encoder_lcca_model1, self).__init__()

        model = ClassifierPneumonia()
        self.latent_dim = 2

        self.encoder = model
        

    def forward(self, x):
        self.encoder.eval()
        h = self.encoder(x)
        #print("hshape",torch.max(h,dim=1).shape)
        h = (h == torch.max(h, dim=1, keepdim=True).values).float()
        #print(h)
        # o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        o = ModelOutput(embedding = h)

        return o
    
class fake_encoder_lcca_model2(nn.Module):

    def __init__(self):
        super(fake_encoder_lcca_model2, self).__init__()

        model = ClassifierBLOOD()
        model.load_state_dict(torch.load('../experiments/classifiers_medmnist/model_2.pt'))
        self.latent_dim = 2

        self.encoder = model
        

    def forward(self, x):
        self.encoder.eval()
        h = self.encoder(x)
        h = (h == torch.max(h, dim=1, keepdim=True).values).float()
        #print(h)

        # o = ModelOutput(embedding = result.float()[:,:self.latent_dim])
        o = ModelOutput(embedding = h)

        return o

def load_fake_dcca_medmnist():
    model1 = fake_encoder_lcca_model1()
    model2 = fake_encoder_lcca_model2()
    return [model1, model2]

##############################################################################################################
# Load the fashion mnist dataset and train



if __name__ == '__main__':
    
    

    # tx = transforms.Compose([transforms.ToTensor(), add_channels()])  # Mnist is already with values between 0 and 1
    
    test_set_p = PneumoniaMNIST('test', transforms.ToTensor(), as_rgb=False)
    print(len(test_set_p))
    test_loader_p = DataLoader(test_set_p,
                             batch_size=2000, shuffle=False)
    
    def transform_blood_labels(targets):
        targets[targets == 1] = 0
        targets[targets == 6] = 1
        return targets.squeeze()
    
    # Train the Blood classifier
    train_set_b = BloodMNIST('train', transforms.ToTensor(), target_transform=transform_blood_labels)
    val_set_b = BloodMNIST('val', transforms.ToTensor(), target_transform=    transform_blood_labels)
    test_set_b = BloodMNIST('test', transforms.ToTensor(), target_transform=  transform_blood_labels)


    # Only keep classes 6 and 1
    selected_classes_t = np.argwhere((train_set_b.labels.squeeze()==6) + (train_set_b.labels.squeeze()==1)).squeeze()
    selected_classes_v = np.argwhere((val_set_b.labels.squeeze()==6) + (val_set_b.labels.squeeze()==1)).squeeze()
    selected_classes_te = np.argwhere((test_set_b.labels.squeeze()==6) + (test_set_b.labels.squeeze()==1)).squeeze()
    
    train_set_b = ResampleDataset(train_set_b, lambda d,i : selected_classes_t[i], size = len(selected_classes_t))
    val_set_b = ResampleDataset(val_set_b, lambda d,i : selected_classes_v[i], size = len(selected_classes_v))
    test_set_b = ResampleDataset(test_set_b, lambda d,i : selected_classes_te[i], size = len(selected_classes_te))
    
    device = 'cuda'
    train_loader_b = DataLoader(train_set_b,256)
    val_loader_b = DataLoader(val_set_b,2000)

    model_b = ClassifierBLOOD().to(device)
    model_p = ClassifierPneumonia().to(device)

    optimizer = optim.Adam(model_b.parameters(),lr = 1e-3)
    objective = nn.CrossEntropyLoss(reduction='sum')
    
        
    def train(train_loader, model, epoch,objective, optimizer):
        model.to(device)
        train_loss = 0.0
        train_acc = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data, labels = data[0].to(device), data[1].to(device)
            logits = model(data)
            loss = objective(logits, labels.squeeze())

            train_acc += accuracy(logits,labels.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print('Epoch {} : train loss : {} , acc {}'.format(epoch, train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset)))
            
            
            

    def test(test_loader, model):
        model.eval()
        loss, acc = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.to('cpu')
                data, labels = data[0], data[1].squeeze()

                logits = model(data)
                acc += accuracy(logits, labels)

        print(
            f"====>  Accuracy on the test set {acc / len(test_loader.dataset)}")
        return 

    


    test(test_loader_p, model_p)

    for epoch in range(30):
        train(train_loader_b, model_b,epoch,objective, optimizer )
        test(val_loader_b,model_b)
    
    torch.save(model_b.state_dict(), '../experiments/classifiers_medmnist/model_2.pt')

