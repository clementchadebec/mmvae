# Define a CNN classifier for MNIST and FASHION-MNIST to use to analyse samples of JMVAE
from pathlib import Path
import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
import datetime
from tqdm import tqdm

def accuracy(logits, labels):

    return torch.sum(labels == torch.argmax(logits, dim=1))

def BinLabels(targets):
    bin_labels = torch.zeros((targets.shape[0], 10))
    for i,c in enumerate(targets):
        bin_labels[i,c] = 1
    return bin_labels

class SVHNClassifier(nn.Module) :

    def __init__(self):
        super().__init__()

        self.input_size = (3,32,32)
        self.num_classes = 10

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,4,1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,4,1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,4,1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.convlayers = nn.Sequential(self.conv1, self.conv2,self.conv3)

        self.fc = nn.Sequential(
              nn.Linear(67712,1024),
              nn.BatchNorm1d(1024),
              nn.Dropout(),
              nn.Linear(1024, 512),
              nn.BatchNorm1d(512),
              nn.Dropout(),
              nn.Linear(512, 10)
        )

    def forward(self,x):
        h = self.convlayers(x)
        f = self.fc(h.reshape(x.shape[0], -1))

        return f

class SVHN_DL():

    def __init__(self, data_path = '../data'):
        self.data_path = data_path
        return

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda', transform=transforms.ToTensor()):
        kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

        train = DataLoader(datasets.SVHN(self.data_path, split='train', download=True, transform=transform),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(datasets.SVHN(self.data_path, split='test', download=True, transform=transform),
                          batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

def load_pretrained_svhn(device='cuda'):
    classifier = SVHNClassifier()
    path = '../experiments/classifier_svhn/2022-06-16/model_8.pt'

    classifier.load_state_dict(torch.load(path))
    classifier.eval()
    classifier.to(device)
    return classifier

##############################################################################################################
# Load the fashion mnist dataset and train



if __name__ == '__main__':

    tx = transforms.ToTensor()  # Mnist is already with values between 0 and 1
    batch_size = 256
    shuffle = True

    train_loader, test_loader = SVHN_DL().getDataLoaders(batch_size, shuffle, tx)

    item, label = next(iter(train_loader))

    model = SVHNClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    objective = torch.nn.BCEWithLogitsLoss(reduction='sum')


    def train(epoch):
        model.train()
        b_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
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


    output_path = Path('../experiments/classifier_svhn' + '/' + datetime.date.today().isoformat() +'/')
    output_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(15):
        train(epoch)
        test(epoch)
        torch.save(model.state_dict(), Path.joinpath(output_path, f'model_{epoch}.pt'))





