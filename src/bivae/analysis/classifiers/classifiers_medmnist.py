# Define a CNN classifier for MNIST and FASHION-MNIST to use to analyse samples of JMVAE
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
from medmnist import PathMNIST, TissueMNIST
from torchvision.models import resnet18, ResNet18_Weights
from bivae.utils import add_channels
from tqdm import tqdm
from bivae.models.nn.medmnist_classifiers import ResNet18



def BinLabels(targets, nb_classes):
    bin_labels = torch.zeros((targets.shape[0], nb_classes))
    for i,c in enumerate(targets):
        bin_labels[i,c] = 1
    return bin_labels

class Classifier1(nn.Module) :

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        
        self.encoder = Encoder_ResNet_AE_medmnist(BaseAEConfig(input_size, num_classes))

    def forward(self,x):
        h = self.encoder(x).embedding
        return h

class Classifier2(nn.Module):
    
    def __init__(self, input_size, num_classes) -> None:
        super().__init__()
        print('loading weights')
        weights = torch.load('../medmnist_models/resnet18_28_1_path.pth')
        self.network = ResNet18(3,9)
        # self.network = resnet18(pretrained = False, num_classes=9)
        self.network.load_state_dict(weights['net'])
        # self.network.fc = nn.Linear(512,num_classes)
        
    def forward(self, x):
        self.network.eval()
        return self.network(x)




##############################################################################################################
# Load the fashion mnist dataset and train



if __name__ == '__main__':
    
    

    # tx = transforms.Compose([transforms.ToTensor(), add_channels()])  # Mnist is already with values between 0 and 1
    tx = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    train_dataset = PathMNIST('train', tx)
    val_dataset = PathMNIST('test', tx)
    num_classes = 9

    batch_size = 256
    num_epochs = 1
    shuffle = True
    train_loader = DataLoader(train_dataset
                              , batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(val_dataset,
                             batch_size=batch_size, shuffle=False)
    print("creating model")
    model = Classifier2((3,28,28), num_classes)
    # model = ResNet18(1,num_classes)
    print("setting the optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    objective = torch.nn.BCEWithLogitsLoss(reduction='sum')


    def train(epoch):
        model.train()
        b_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            data, labels = data[0], data[1].squeeze()
            logits = model(data)
            loss = objective(logits, BinLabels(labels, num_classes))
            loss.backward()
            optimizer.step()
            b_loss += loss.item()

        print(f"====> Epoch {epoch} Train Loss {b_loss / len(train_loader.dataset)}")


    def test(epoch):
        model.eval()
        loss, acc = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data, labels = data[0], data[1].squeeze()
                logits = model(data)
                bloss = objective(logits, BinLabels(labels, num_classes))
                acc += accuracy(logits, labels)
                loss += bloss.item()

        print(
            f"====> Epoch {epoch} Test Loss {loss / len(test_loader.dataset)} Accuracy {acc / len(test_loader.dataset)}")
        return loss/(len(test_loader.dataset))

    output_path = Path('../experiments/classifier_' + 'tissue_mnist' + '/')
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = torch.inf
    for epoch in tqdm(range(num_epochs)):
        # train(epoch)
        test_loss = test(epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), Path.joinpath(output_path, f'model.pt'))





