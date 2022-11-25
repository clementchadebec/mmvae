
from datetime import datetime
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import json

import torch
import wandb
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler


from bivae.dcca.linear_cca import linear_cca
from bivae.dataloaders import CELEBA_DL
from bivae.dcca.models import DeepCCA_celeba
from bivae.dcca.utils import  svm_classify_view, unpack_data, visualize_umap, save_encoders

torch.set_default_tensor_type(torch.FloatTensor)



class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        print('Solver initialized')

    def fit(self, train_loader, val_loader, tx1=None, tx2=None, checkpoint=None):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        print('Starting the optimization')


        if val_loader is not None:
            best_val_loss = 0
            
        num_epochs_without_improvement = 0
        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()

            for dataT in tqdm(train_loader):
                data = unpack_data(dataT)
                self.optimizer.zero_grad()
                batch_x1 = data[0]
                batch_x2 = data[1]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            wandb.log({'train_loss' : train_loss})
            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if val_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(val_loader)[0]
                    wandb.log({'val_loss' : val_loss})
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        print(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        save_encoders(model,checkpoint)
                        num_epochs_without_improvement = 0
                    else:
                        print("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
                        num_epochs_without_improvement+=1
            else:
                save_encoders(model, checkpoint)
            epoch_time = time.time() - epoch_start_time
            print((info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss)))
            
            if num_epochs_without_improvement ==10:
                break
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(train_loader)
            self.train_linear_cca(outputs[0], outputs[1])


        if val_loader is not None:
            loss = self.test(val_loader)[0]
            print("loss on validation data: {:.4f}".format(loss))

        

    def test(self, test_loader, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(test_loader)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs_cca = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), [*outputs_cca,outputs[-1]]
            else:
                return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            data_size = len(train_loader.dataset)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            labels = []
            for dataT in test_loader:
                labels.append(dataT[0][1])
                data = unpack_data(dataT)
                batch_x1 = data[0]
                batch_x2 = data[1]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy(),
                   torch.cat(labels, dim=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':

    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
    parser.add_argument('--config-path', type=str, default='')


    # args
    info = parser.parse_args()

    ############
    # Parameters Section
    with open(info.config_path, 'r') as fcc_file:
        args = argparse.Namespace()
        args.__dict__.update(json.load(fcc_file))

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the models
    save_to = Path('../experiments/dcca/celeba/')
    save_to.mkdir(parents=True, exist_ok=True)

    # the size of the new space learned by the model (number of the new features)
    outdim_size = args.outdim_size_dcca


    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = args.num_epochs_dcca
    batch_size = 800
    train_loader,test_loader, val_loader = CELEBA_DL(args.data_path).getDataLoaders(batch_size=batch_size)


    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True
    # end of parameters section
    ############
    
    wandb.init(project = 'DCCA_celeba', entity = 'asenellart', config = {'batch_size' : batch_size,
                                                                            'learning_rate': learning_rate,
                                                                            'reg_par' : reg_par,
                                                                            'linear_cca' : linear_cca is not None,
                                                                            'outdim_size' : outdim_size,
                                                                            'num_epochs': epoch_num},
                   dir=str(save_to) + '/wandb')
    
    # Save parameters of training
    with open('{}/args.json'.format(save_to), 'w') as fp:
        json.dump(args.__dict__, fp)


    # Building, training, and producing the new features by DCCA
    model = DeepCCA_celeba(outdim_size, use_all_singular_values, device=device)
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    solver.fit(train_loader, val_loader, checkpoint=save_to)

    # Save parameters for the lcca
    if apply_linear_cca :
        np.save(str(save_to) + '/l_cca_w.npy', l_cca.w)
        np.save(str(save_to) + '/l_cca_m.npy', l_cca.m)
        np.save(str(save_to) + '/l_cca_D.npy', l_cca.D)
    
    # Plot the Singular values of the DCCA
    plt.plot(l_cca.D)
    plt.savefig(str(save_to) + '/singular_values.png')
    plt.close()

   
