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


def BinLabels(targets):
    bin_labels = torch.zeros((targets.shape[0], 10))
    for i,c in enumerate(targets):
        bin_labels[i,c] = 1
    return bin_labels

class MnistClassifier(nn.Module) :

    def __init__(self):
        super().__init__()

        self.input_size = (1,28,28)
        self.num_classes = 10

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,4,1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,4,1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.fc = nn.Sequential(
              nn.Linear(30976,512),
              nn.Dropout(),
              nn.Linear(512, 10)

        )

    def forward(self,x):
        h = self.conv1(x)
        h2 = self.conv2(h)
        f = self.fc(h2.reshape(x.shape[0], -1))

        return f


def load_pretrained_mnist(path = '../experiments/classifier_numbers/model.pt', device = 'cuda'):
    # Define the classifiers for analysis
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(path))
    classifier.eval()
    classifier.to(device)
    return classifier

def load_pretrained_fashion(path = '../experiments/classifier_fashion/model.pt', device='cuda'):
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(path))
    classifier.eval()
    classifier.to(device)
    return classifier


##############################################################################################################
# Load the fashion mnist dataset and train



if __name__ == '__main__':
    parser = argparse.ArgumentParser('classifier_mnist')
    parser.add_argument('--mnist-type', type=str, default='numbers')
    mnist_type = parser.parse_args().mnist_type

    tx = transforms.ToTensor()  # Mnist is already with values between 0 and 1
    batch_size = 256
    num_epochs = 15
    shuffle = True
    dataset = datasets.FashionMNIST if mnist_type == 'fashion' else datasets.MNIST
    print("loading datasets")
    train_loader = DataLoader(dataset('../data', train=True, download=True, transform=tx)
                              , batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset('../data', train=False, download=True, transform=tx),
                             batch_size=batch_size, shuffle=False)
    print("creating model")
    model = MnistClassifier()
    print("setting the optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    objective = torch.nn.BCEWithLogitsLoss(reduction='sum')


    def train(epoch):
        model.train()
        b_loss = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data, labels = data[0], data[1]
            logits = model(data)
            loss = objective(logits, BinLabels(labels))
            loss.backward()
            optimizer.step()
            b_loss += loss.item()

        print(f"====> Epoch {epoch} Train Loss {b_loss / len(train_loader.dataset)}")


    def test(epoch):
        model.eval()
        loss, acc = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data, labels = data[0], data[1]
                logits = model(data)
                bloss = objective(logits, BinLabels(labels))
                acc += accuracy(logits, labels)
                loss += bloss.item()

        print(
            f"====> Epoch {epoch} Test Loss {loss / len(test_loader.dataset)} Accuracy {acc / len(test_loader.dataset)}")
        return loss/(len(test_loader.dataset))

    output_path = Path('../experiments/classifier_' + mnist_type + '/')
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = torch.inf
    for epoch in range(num_epochs):
        train(epoch)
        test_loss = test(epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), Path.joinpath(output_path, f'model.pt'))





