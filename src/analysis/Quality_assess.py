""" Define a class for assessing the quality of a generative model using
FID and PRD analysis with a defined encoder and parameters"""

import numpy as np
from tqdm import tqdm
from utils import unpack_data
import torch
import analysis.prd as prd
from analysis.pytorch_fid import calculate_fid_from_embeddings
from torch import nn
from utils import adjust_shape
from dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader
from analysis.pytorch_fid import InceptionV3, get_activations


class GenerativeQualityAssesser():

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

        self.ref_activations = self.get_activations(ref_dataloader)
        # self.check_activations()

    def get_activations(self, dataloader):
        for encoder in self.encoders:
            encoder.eval()
        pred_arr = [np.empty((self.n_samples,d)) for d in self.dims]

        start_idx = 0

        for i, batch in enumerate(tqdm(dataloader)):
            if i == self.nb_batches:
                break
            batch = unpack_data(batch, device=self.device)
            for m in range(len(self.encoders)):
                with torch.no_grad():
                    pred = self.encoders[m](batch[m]) # batchsize x dims[m]

                pred_arr[m][start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        pred_arr = np.concatenate(pred_arr, axis=1)
        print("pred_arr shape ", pred_arr.shape)
        return pred_arr


    def compute_fid_prd(self, gen_dataloader, compute_unimodal=False):
        """
                Compute the prd data between the gen_data given as input and the reference data that is the test dataloader.
                """

        # Defines the transformations to format the images for the Inception network

        # Compute embeddings
        gen_act = self.get_activations(gen_dataloader)

        # Compute prd
        prd_data = prd.compute_prd_from_embedding(self.ref_activations, gen_act, self.nb_clusters)

        # Compute fid
        fid = calculate_fid_from_embeddings(self.ref_activations, gen_act)


        if compute_unimodal:
            prd_data0 = prd.compute_prd_from_embedding(self.ref_activations[:, :2048], gen_act[:, :2048],
                                                       num_clusters=self.nb_clusters)
            prd_data1 = prd.compute_prd_from_embedding(self.ref_activations[:, 2048:], gen_act[:, 2048:],
                                                       num_clusters=self.nb_clusters)
            fid0 = calculate_fid_from_embeddings(self.ref_activations[:, :2048], gen_act[:, :2048])
            fid1 = calculate_fid_from_embeddings(self.ref_activations[:, 2048:], gen_act[:, 2048:])

            return fid, prd_data, fid0, prd_data0, fid1, prd_data1

        return fid, prd_data

    def check_activations(self):
        # Sanity check :
        print('Fid distance with itself ', calculate_fid_from_embeddings(self.ref_activations, self.ref_activations))

        # Define the model to compute the embeddings
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(self.device)

        # Compute embeddings
        ref_activations = get_activations(self.ref_data, model, dims=2048, nb_batches=self.nb_batches)
        print(ref_activations[:2], self.ref_activations[:2])
        1/0
    def GenerateDataloader(self, gen_data, transform):

        # Create a dataloader with the formatted generated data
        data = torch.stack(adjust_shape(gen_data[0], gen_data[1]))
        dataset = MultimodalBasicDataset(data, transform)
        return DataLoader(dataset, batch_size=self.batchsize, shuffle=True)








