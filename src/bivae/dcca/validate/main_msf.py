""" Import DCCA encoders, compute all outputs for the validation set and try to fit an svm for each attribute
We also observe the singular values of the DCCA to see the total correlation and if it is worth shortening the embedding"""


from bivae.dcca.models.celeba import load_dcca_celeba
from bivae.dataloaders import MNIST_SVHN_FASHION_DL
import torch
from bivae.utils import unpack_data
from bivae.dcca.utils import svm_classify_view
from bivae.dcca.models.mnist_svhn_fashion import DeepCCA_MNIST_SVHN_FASHION
from tqdm import tqdm

model = DeepCCA_MNIST_SVHN_FASHION(16,False, 'cuda')
model.load_state_dict('../experiments/dcca_msf/model.pt')
1/0
train, test, val = MNIST_SVHN_FASHION_DL().getDataLoaders(batch_size = 128)

def _get_outputs(dcca, test_loader):
        with torch.no_grad():
            dcca[0].eval()
            dcca[1].eval()
            dcca[0].cuda()
            dcca[1].cuda()
            data_size = len(test_loader.dataset)
            
            losses = []
            outputs1 = []
            outputs2 = []
            labels = []
            for dataT in tqdm(test_loader):
                labels.append(dataT[0][1])
                data = unpack_data(dataT)
                batch_x1 = data[0]
                batch_x2 = data[1]
                o1, o2 = dcca[0](batch_x1), dcca[1](batch_x2)
                outputs1.append(o1.embedding)
                outputs2.append(o2.embedding)

        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy(),
                   torch.cat(labels, dim=0).cpu().numpy()]
        return outputs

train_outputs = _get_outputs(dcca, train)
test_outputs = _get_outputs(dcca, test)

for i in range(40):
    """train a linear svc to classify each attribute"""
    train_o  = [train_outputs[0], train_outputs[1], train_outputs[2][:,i]]
    test_o  = [test_outputs[0], test_outputs[1], test_outputs[2][:,i]]
    test_acc = svm_classify_view(train_o, test_o, C=1)
    print(f'Attribute {i} : {test_acc*100}')
    
    

