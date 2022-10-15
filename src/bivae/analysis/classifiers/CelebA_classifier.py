""" Train a CelebA classifier to analyse the results of the generative models"""

import torch
from torch import optim
from pathlib import Path
import datetime
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_AE_CELEBA, BaseAEConfig
from bivae.dataloaders import CELEBA_DL
import numpy as np

from bivae.utils import unpack_data

# We use the same architecture as for the encoder as a classifier

class Resnet_classifier_celeba(torch.nn.Module):

    def __init__(self):
        super(Resnet_classifier_celeba, self).__init__()
        self.encoder=Encoder_ResNet_AE_CELEBA(BaseAEConfig(latent_dim = 40))

    def forward(self, x):

        logits = self.encoder(x)['embedding']
        return logits

def accuracy(logits, labels):
    return torch.sum((logits > 0).int() == labels)/(np.prod(labels.shape))


if __name__ == '__main__':


    batch_size = 256
    shuffle = True

    train_loader, test_loader, val_loader = CELEBA_DL('../data/').getDataLoaders(batch_size,shuffle)

    model = Resnet_classifier_celeba().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    objective = torch.nn.BCEWithLogitsLoss(reduction='sum')


    def train(epoch):
        model.train()
        b_loss = 0
        for i, data_ in enumerate(train_loader):
            data = unpack_data(data_)
            optimizer.zero_grad()
            data, labels = data[0], data[1] # data is of shape (n_batch,3, 64, 64) and labels (n_batch, 1, 1, 40)
            logits = model(data) # Shape (n_batch, 40)
            loss = objective(logits, labels.squeeze())
            loss.backward()
            optimizer.step()
            b_loss += loss.item()

        print(f"====> Epoch {epoch} Train Loss {b_loss / len(train_loader.dataset)}")


    def test(epoch):
        model.eval()
        loss, acc = 0, 0
        with torch.no_grad():
            for i, data_ in enumerate(val_loader):
                data = unpack_data(data_)
                data, labels = data[0], data[1]
                logits = model(data)
                bloss = objective(logits, labels.squeeze())
                acc += accuracy(logits, labels)
                loss += bloss.item()


        test_loss = loss / len(val_loader.dataset)
        print(
            f"====> Epoch {epoch} Test Loss {test_loss} Accuracy {acc / len(val_loader.dataset)}")
        return test_loss

    output_path = Path('../experiments/classifier_celeba/' + datetime.date.today().isoformat() +'/')
    output_path.mkdir(parents=True, exist_ok=True)

    best_test_loss = np.inf
    for epoch in range(50):
        train(epoch)
        test_loss = test(epoch)
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), Path.joinpath(output_path, f'model.pt'))
            best_test_loss=test_loss







