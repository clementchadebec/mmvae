import torch
from torchvision import datasets, transforms

'''Make associations between three datasets MNIST-SVHN-FASHION'''

def rand_match_on_idx(l1, idx1, l2, idx2,l3, idx3, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 , _idx3 = [], [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2, l_idx3 = idx1[l1 == l], idx2[l2 == l], idx3[l3==l]
        n = min(l_idx1.size(0), l_idx2.size(0), l_idx3.size(0), max_d)
        l_idx1, l_idx2, l_idx3 = l_idx1[:n], l_idx2[:n], l_idx3[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
            _idx3.append(l_idx3[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2), torch.cat(_idx3)

if __name__ == '__main__':
    max_d = 10000  # maximum number of datapoints per class
    dm = 5       # data multiplier: random permutations to match

    # get the individual datasets
    tx = transforms.ToTensor()
    path_data = '/home/agathe/Code/datasets'
    train_mnist = datasets.MNIST(path_data, train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST(path_data, train=False, download=True, transform=tx)
    train_svhn = datasets.SVHN(path_data, split='train', download=True, transform=tx)
    test_svhn = datasets.SVHN(path_data, split='test', download=True, transform=tx)
    train_fashion = datasets.FashionMNIST(path_data, train=True, download=True, transform=tx )
    test_fashion = datasets.FashionMNIST(path_data, train=False, download=True, transform=tx )

    # svhn labels need extra work
    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

    mnist_l, mnist_li = train_mnist.targets.sort()
    svhn_l, svhn_li = train_svhn.labels.sort()
    f_l, f_li = train_fashion.targets.sort()
    
    idx1, idx2, idx3 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, f_l, f_li, max_d=max_d, dm=dm)
    print('len train idx:', len(idx1), len(idx2), len(idx3))
    torch.save(idx1, '../data/train-msf-mnist-idx.pt')
    torch.save(idx2, '../data/train-msf-svhn-idx.pt')
    torch.save(idx3, '../data/train-msf-fashion-idx.pt' )

    mnist_l, mnist_li = test_mnist.targets.sort()
    svhn_l, svhn_li = test_svhn.labels.sort()
    f_l, f_li = test_fashion.targets.sort()
    idx1, idx2, idx3 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li,f_l, f_li, max_d=max_d, dm=dm)
    shuffle = torch.randperm(len(idx1))
    print('len test idx:', len(idx1), len(idx2), len(idx3))
    torch.save(idx1[shuffle], '../data/test-msf-mnist-idx.pt')
    torch.save(idx2[shuffle], '../data/test-msf-svhn-idx.pt')
    torch.save(idx3[shuffle], '../data/test-msf-fashion-idx.pt')

