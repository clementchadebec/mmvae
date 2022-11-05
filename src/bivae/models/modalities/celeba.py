''' Function for CelebA dataset that distributes over the different celeba models'''

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

from bivae.dataloaders import BasicDataset
from bivae.utils import unpack_data, add_channels
from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance

def compute_accuracies(model, data, runPath, epoch, classes, n_data=100, ns=300, freq=10):
    """
    Celeba special compute accuracies since each image has a vector of attribute 
    and not one labels. 

    Args:
        data (list of Tensors): the data to use to compute conditional accuracies (list of Tensors)
        runPath (str or Path): The path to eventually save results
        epoch (int): epoch
        classes (list or array): multidimensional labels
        n_data (int): How much of the data to use. Defaults to 100.
        ns (int, optional): How much sample to generate conditionally. Defaults to 300.
        freq (int, optional): _description_. Defaults to 10.

    Returns:
        Dict of metrics. Accuracies.
    """



    bdata = [d[:n_data] for d in data]
    samples = model._sample_from_conditional(bdata, n=ns)
    cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

    # Compute the labels
    preds2 = model.classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod2))  # 8*n x 40
    labels2 = (preds2 > 0).int().reshape(n_data, ns,40)

    preds1 = model.classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod1))  # 8*n x 10
    labels1 = (preds1 > 0).int().reshape(n_data, ns, 40)
    classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1, 0,2).cuda()
    print(classes_mul.shape)

    acc2 = torch.sum(classes_mul == labels2) / (n_data * ns*40)
    acc1 = torch.sum(classes_mul == labels1) / (n_data * ns*40)

    metrics = dict(accuracy1=acc1, accuracy2=acc2)

    # Compute the joint accuracy
    data = model.generate('', 0, N=ns, save=False)
    labels_celeb = model.classifier1(data[0]) > 0
    labels_attributes = model.classifier2(data[1]) > 0

    joint_acc = torch.sum(labels_attributes == labels_celeb) / (ns * 40)
    metrics['joint_coherence'] = joint_acc

    
    return metrics



def compute_fid_celeba(model, batch_size):

        # Define the inception model used to compute FID
        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test,_ = model.getDataLoaders(batch_size,transform=tx)

        ref_activations = []

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations.append(model(data[0]))

        ref_activations = np.concatenate(ref_activations)

        # Generate data from conditional

        _, test,_ = model.getDataLoaders(batch_size)

        gen_samples = []
        for dataT in test:
            data=unpack_data(dataT)
            gen = model._sample_from_conditional(data, n=1)[1][0]


            gen_samples.extend(gen)

        gen_samples = torch.cat(gen_samples).squeeze()
        # print(gen_samples.shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_dataset = BasicDataset(gen_samples,transform=tx)
        gen_dataloader = DataLoader(gen_dataset,batch_size=batch_size)

        gen_activations = []
        for data in gen_dataloader:
            gen_activations.append(model(data[0]))
        gen_activations = np.concatenate(gen_activations)

        # print(ref_activations.shape, gen_activations.shape)

        mu1, mu2 = np.mean(ref_activations, axis=0), np.mean(gen_activations, axis=0)
        sigma1, sigma2 = np.cov(ref_activations, rowvar=False), np.cov(gen_activations, rowvar=False)

        # print(mu1.shape, sigma1.shape)

        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # print(fid)
        return {'fid' : fid}


 


   










