import pathlib

import numpy as np
from tqdm import tqdm
import time
from pathlib import Path


import torch
import wandb
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler


from linear_cca import linear_cca
from dataloaders import MNIST_SVHN_DL
from DeepCCAModels import DeepCCA_MNIST_SVHN
from utils import load_data, svm_classify_svhn, unpack_data, visualize_umap, save_encoders

torch.set_default_tensor_type(torch.DoubleTensor)



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

        wandb.init(project = 'DCCA_mnist_svhn', entity = 'asenellart', config = {'batch_size' : batch_size,
                                                                                 'learning_rate': learning_rate,
                                                                                 'reg_par' : reg_par,
                                                                                 'linear_cca' : linear_cca is not None},
                   mode = 'online')
        print('Solver initialized')

    def fit(self, train_loader, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint=None):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        print('Starting to optimization')

        data_size = len(train_loader.dataset)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1 = vx1.to(self.device)
            vx2 = vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1 = tx1.to(self.device)
            tx2 = tx2.to(self.device)

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

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        print(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        save_encoders(model,checkpoint)
                    else:
                        print("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                save_encoders(model, checkpoint)
            epoch_time = time.time() - epoch_start_time
            print((info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss)))
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(train_loader)
            self.train_linear_cca(outputs[0], outputs[1])


        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            print("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            print('loss on test data: {:.4f}'.format(loss))

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


    ############
    # Parameters Section

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the models
    save_to = Path('./')
    save_to.mkdir(parents=True, exist_ok=True)

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 15


    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 10
    batch_size = 800
    train_loader,test_loader = MNIST_SVHN_DL('/home/agathe/Code/vaes/mmvae/data').getDataLoaders(batch_size=batch_size)


    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False
    # end of parameters section
    ############


    # Building, training, and producing the new features by DCCA
    model = DeepCCA_MNIST_SVHN(outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    solver.fit(train_loader, checkpoint=save_to)
    # TODO: Save l_cca model if needed


    # Training and testing of SVM with linear kernel on the view 1 with new features
    losses, outputs_t =  solver.test(train_loader, use_linear_cca=apply_linear_cca)
    losses, outputs_s = solver.test(test_loader, use_linear_cca=apply_linear_cca)
    test_acc = svm_classify_svhn(outputs_t, outputs_s, C=0.01)
    wandb.log({'embedding_svhn' : visualize_umap(outputs_s[1], outputs_s[2])})
    wandb.log({'embedding_mnist' : visualize_umap(outputs_s[0], outputs_s[2])})
    print("Accuracy on view svhn (test data) is:", test_acc*100.0)
    # Saving new features in a gzip pickled file specified by save_to

