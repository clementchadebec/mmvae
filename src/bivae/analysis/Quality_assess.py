""" Define a class for assessing the quality of a generative model using
FID and PRD analysis with a defined encoder and parameters"""

import numpy as np
from tqdm import tqdm
from bivae.utils import unpack_data, add_channels, adjust_shape
import torch
from bivae.analysis import prd
from bivae.analysis.pytorch_fid import calculate_fid_from_embeddings
from torch import nn
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from bivae.analysis.pytorch_fid.inception import wrapper_inception
from bivae.analysis.pytorch_fid.custom_encoders import wrapper_pythae_model
import pythae
from pythae.models import AutoModel
from umap import UMAP
from bivae.vis import plot_embeddings

class GenerativeQualityAssesser():
    gen_transform = None  # to be defined in each subclass

    def __init__(self, encoders,batchsize, n_samples,nb_clusters, ref_dataloader, dims, device ='cuda'):
        """
        Make sure n_samples is a multiple of batchsize
        """
        self.encoders = encoders
        self.nb_clusters = nb_clusters # in PRD
        self.ref_data = ref_dataloader
        self.batchsize = batchsize
        self.n_samples = n_samples
        self.dims=dims
        self.device = device
        self.nb_batches = self.n_samples // batchsize
        # Compute the reference activations

        self.ref_activations, self.ref_labels = self.get_activations(ref_dataloader)
        # self.check_activations()


    def check_activations(self, runpath, data=None, labels=None):
        """ Compute a u-map representation of the activations and see if the classes are separated """
        if data is None:
            data =self.ref_activations
            labels = self.ref_labels

        act0 = data[:,:self.dims[0]]
        act1 = data[:,self.dims[0]:]
        umap_emb = UMAP(n_neighbors=40, transform_seed=torch.initial_seed())
        act0 = umap_emb.fit_transform(act0)
        act1 = umap_emb.fit_transform(act1)
        act_joint = umap_emb.fit_transform(data)

        plot_embeddings(act0,labels[0], np.unique(labels[0]), f'{runpath}/check_act0.png')
        plot_embeddings(act1,labels[1], np.unique(labels[1]), f'{runpath}/check_act1.png')
        plot_embeddings(act_joint,labels[1], np.unique(labels[1]), f'{runpath}/check_act_joint.png')




    def get_activations(self, dataloader):
        for encoder in self.encoders:
            encoder.eval()
        pred_arr = [np.empty((self.n_samples,d)) for d in self.dims]
        # labels = [np.empty(self.n_samples) for _ in self.dims]
        start_idx = 0

        for i, batch in enumerate(tqdm(dataloader)):
            if i == self.nb_batches:
                break
            classes = [batch[0][1], batch[1][1]]
            batch = unpack_data(batch, device=self.device)
            for m in range(len(self.encoders)):
                with torch.no_grad():
                    pred = self.encoders[m](batch[m]) # batchsize x dims[m]

                pred_arr[m][start_idx:start_idx + pred.shape[0]] = pred
                # labels[m][start_idx:start_idx + pred.shape[0]] = classes[m]

            start_idx = start_idx + pred.shape[0]

        pred_arr = np.concatenate(pred_arr, axis=1)
        print("pred_arr shape ", pred_arr.shape)
        # return pred_arr, labels
        return pred_arr, None

    def compute_fid_prd(self, gen_dataloader, runPath, compute_unimodal=False):
        """
                Compute the prd data between the gen_data given as input and the reference data that is the test dataloader.
                """

        # Defines the transformations to format the images for the Inception network

        # Compute embeddings
        gen_act, gen_labels = self.get_activations(gen_dataloader)
        concat_activations = np.concatenate([self.ref_activations,gen_act])
        # Check superposition of real data and gen_data in embedding space
        self.check_activations(runPath,concat_activations,
                               [np.concatenate([np.ones(len(self.ref_activations)), np.zeros(len(gen_act))])]*2)
        # Compute prd
        prd_data = prd.compute_prd_from_embedding(self.ref_activations, gen_act, self.nb_clusters)

        # Compute fid
        fid = calculate_fid_from_embeddings(self.ref_activations, gen_act)

        results_dict = dict(fid=fid,prd_data=prd_data, concat_activations=concat_activations)

        if compute_unimodal:
            prd_data0 = prd.compute_prd_from_embedding(self.ref_activations[:, :self.dims[0]], gen_act[:, :self.dims[0]],
                                                       num_clusters=self.nb_clusters)
            prd_data1 = prd.compute_prd_from_embedding(self.ref_activations[:, self.dims[0]:], gen_act[:, self.dims[0]:],
                                                       num_clusters=self.nb_clusters)
            fid0 = calculate_fid_from_embeddings(self.ref_activations[:, :self.dims[0]], gen_act[:, :self.dims[0]])
            fid1 = calculate_fid_from_embeddings(self.ref_activations[:, self.dims[0]:], gen_act[:, self.dims[0]:])

            results_dict['fid0'] = fid0
            results_dict['fid1'] = fid1
            results_dict['prd_data0'] = prd_data0
            results_dict['prd_data1'] = prd_data1

        return results_dict

    def GenerateDataloader(self, gen_data, transform):

        # Create a dataloader with the formatted generated data
        data = torch.stack(adjust_shape(gen_data[0], gen_data[1]))
        dataset = MultimodalBasicDataset(data, transform)
        return DataLoader(dataset, batch_size=self.batchsize, shuffle=True)



class Inception_quality_assess(GenerativeQualityAssesser):

    batchsize = 50
    n_samples = 100*batchsize
    gen_transform = transforms.Compose([transforms.Resize((299, 299)), add_channels()])
    nb_clusters = 10
    dims = [2048,2048]
    name = 'Inception_quality_assess'

    def __init__(self, model):
        encoders = [wrapper_inception(), wrapper_inception()]
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])
        t, s, v = model.getDataLoaders(self.batchsize, transform=tx) # get the test dataset as dataloader
        super().__init__(encoders, self.batchsize, self.n_samples,self.nb_clusters,s,self.dims)




class custom_mnist_fashion(GenerativeQualityAssesser):

    batchsize = 47
    n_samples = 100*batchsize
    gen_transform = None
    nb_clusters = 20
    dims = [16,16]
    name = 'custom_mnist_fashion'
    device = 'cuda'

    def __init__(self, model):
        mnist_vae = AutoModel.load_from_folder(
            '/home/agathe/Code/vaes/benchmark_VAE/my_model/MNIST_cnn_vae_training_22_08/final_model/')
        fashion_vae = AutoModel.load_from_folder(
            '/home/agathe/Code/vaes/benchmark_VAE/my_model/FashionMnist_cnn_vae_training_19_08/final_model/')

        mnist_vae.to(self.device)
        fashion_vae.to(self.device)
        encoders = [wrapper_pythae_model(mnist_vae),wrapper_pythae_model(fashion_vae)]
        tx = transforms.ToTensor()
        t,s,v = model.getDataLoaders(self.batchsize, transform = tx)
        print(torch.max(s.dataset[0][0][0]),torch.max(s.dataset[0][1][0]) )
        super(custom_mnist_fashion, self).__init__(encoders, self.batchsize,self.n_samples,self.nb_clusters,s,self.dims)