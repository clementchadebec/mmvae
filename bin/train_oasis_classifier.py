
import numpy as np
from time import time
from copy import deepcopy
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from bivae.dataloaders import MRIDataset
from bivae.analysis.classifiers.oasis_classifier_train import create_model
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from bivae.utils import add_channels
import wandb
from sklearn.model_selection import StratifiedKFold





class select_slice(object):

    def __init__(self):
        self.eval_mode = False

    def train(self):
        self.eval_mode = False

    def eval(self):
        self.eval_mode = True

    def __call__(self, pic):
        slice = 60 if not self.eval_mode else 60 + np.random.randint(-5,5)
        return pic[:,:,:,60]





def train(model, train_loader, criterion, optimizer, epoch):
    """
    Method used to train a CNN

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """


    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader, 0):
        images, labels = data['image'].cuda(), data['label'].cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    mean_loss = total_loss / len(train_loader.dataset)
    print('Epoch %i: Train loss = %f' % (epoch, mean_loss))
    wandb.log({'train_loss' : mean_loss})

    return


def test(model, data_loader, criterion, epoch):
    """
    Method used to test a CNN

    Args:
        model: (nn.Module) the neural network
        data_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for on the slice level.
        results_metrics: (dict) a set of metrics
    """
    model.eval()
    columns = ["participant_id", "proba0", "proba1",
               "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data['image'].cuda(), data['label'].cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)

            for idx, sub in enumerate(data['participant_id']):
                row = [sub,
                       probs[idx, 0].item(), probs[idx, 1].item(),
                       labels[idx].item(), predicted[idx].item()]
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
    wandb.log({'test_loss' : results_metrics['mean_loss']})
    return results_df, results_metrics

def trainer(model, criterion, n_epochs, train_loader, val_loader):

    best_model = deepcopy(model)
    test_best_loss = np.inf

    for epoch in range(n_epochs):
        train(model, train_loader, criterion, optimizer,epoch)
        _, results_metrics = test(model,val_loader, criterion, epoch)
        test_loss = results_metrics['mean_loss']
        print('Epoch %i: Test loss = %f' % (epoch, test_loss))

        if test_loss < test_best_loss:
            best_model = deepcopy(model)

    return best_model


def compute_metrics(ground_truth, prediction):
    """Computes the accuracy, sensitivity, specificity and balanced accuracy"""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity
    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    else:
        metrics_dict['sensitivity'] = 0.0

    # Specificity
    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)
    else:
        metrics_dict['specificity'] = 0.0

    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

    return metrics_dict


if __name__ == '__main__':

    img_dir = '/home/agathe/Code/datasets/OASIS-1_dataset/preprocessed'
    train_df = pd.read_csv('/home/agathe/Code/datasets/OASIS-1_dataset/tsv_files/lab_1/train_unbalanced.tsv',  sep='\t')
    test_df = pd.read_csv('/home/agathe/Code/datasets/OASIS-1_dataset/tsv_files/lab_1/test_unbalanced.tsv', sep='\t')

    # Create 5 k-folds to correctly evaluate the model performance
    kf = StratifiedKFold(n_splits=5)
    kf.get_n_splits(train_df.index.values, train_df['diagnosis'] == 'AD')

    transf = transforms.Compose([select_slice(), add_channels()])


    test_data = MRIDataset(img_dir, test_df, transform=transf)


    model = create_model().cuda()
    batchsize = 32
    learning_rate = 10 ** -4
    n_epochs = 20

    wandb.init(project='train_oasis_classifier', entity='asenellart',
               config={'lr': learning_rate, 'n_epochs': n_epochs})

    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    all_folds_metrics = []
    for train_index, val_index in kf.split(train_df):
        ftrain = train_df.loc[train_index].reset_index()
        fvalid = train_df.loc[val_index].reset_index()
        train_data = MRIDataset(img_dir, ftrain, transform=transf)
        val_data = MRIDataset(img_dir,fvalid,transform=transf)
        train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=False,num_workers=8)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


        best_model = trainer(model, criterion, n_epochs,train_dataloader, val_dataloader)

        # Test
        result_df, results_metrics = test(best_model,test_loader, criterion, 0)
        all_folds_metrics.append(results_metrics)
        print('Fold test results : ', results_metrics)

    print(all_folds_metrics)