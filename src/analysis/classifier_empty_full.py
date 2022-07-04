# Define a basic empty full classifier for my experiments on circles/squares
from pathlib import Path
import torch.nn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
import datetime
from torchnet.dataset import TensorDataset, ResampleDataset
from torch.utils.data import Dataset
from tqdm import tqdm

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





def accuracy(logits, labels):

    return torch.sum(labels == torch.argmax(logits, dim=1))

def BinLabels(targets):
    bin_labels = torch.zeros((targets.shape[0], 10))
    for i,c in enumerate(targets):
        bin_labels[i,c] = 1
    return bin_labels

class CirclesClassifier(nn.Module) :

    def __init__(self):
        super().__init__()

        self.input_size = (1,32,32)
        self.num_classes = 10



        self.fc = nn.Sequential(
              nn.Linear(32*32,512),
              nn.Dropout(),
              nn.Linear(512, 10)

        )

    def forward(self,x):
        h = x.reshape(*x.shape[:-3], 32*32)

        f = self.fc(h)

        return f






##############################################################################################################
# Load the fashion mnist dataset and train

type = 'circles'

batch_size = 256
shuffle=True
train_loader, test_loader = CIRCLES_DL(type, '/home/agathe/Code/vaes/mmvae/data/circles_squares/').getDataLoaders(batch_size,shuffle)

model = CirclesClassifier()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
objective = torch.nn.BCEWithLogitsLoss(reduction='sum')

def train(epoch):
    model.train()
    b_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        data, labels = data[0], data[1]
        logits = model(data)
        loss = objective(logits,BinLabels(labels))
        loss.backward()
        optimizer.step()
        b_loss += loss.item()

    print(f"====> Epoch {epoch} Train Loss {b_loss/len(train_loader.dataset)}")

def test(epoch):
    model.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data, labels = data[0], data[1]
            logits = model(data)
            bloss = objective(logits, BinLabels(labels))
            acc += accuracy(logits, labels)
            loss +=bloss.item()

    print(f"====> Epoch {epoch} Test Loss {loss/len(test_loader.dataset)} Accuracy {acc/len(test_loader.dataset)}")


if __name__ == '__main__':

    output_path = Path('../experiments/classifier_' + type + '/' + datetime.date.today().isoformat() +'/')
    output_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(5):
        train(epoch)
        test(epoch)
        torch.save(model.state_dict(), Path.joinpath(output_path, f'model_{epoch}.pt'))





