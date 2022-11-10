
import torch
import numpy as np


from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.utils import unpack_data, add_channels
from torchvision import transforms
from bivae.dataloaders import MultimodalBasicDataset
from torch.utils.data import DataLoader


def fid(model, batch_size):

        model_fid = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = model.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[],[]]

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations[0].append(model_fid(data[0]))
            ref_activations[1].append(model_fid(data[1]))

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = model.getDataLoaders(batch_size)

        gen_samples = [[],[]]
        for dataT in test:
            data = unpack_data(dataT)
            gen = model._sample_from_conditional(data, n=1)
            gen_samples[0].extend(gen[1][0])
            gen_samples[1].extend(gen[0][1])

        gen_samples = [torch.cat(g).squeeze(0) for g in gen_samples]
        print(gen_samples[0].shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_dataset = MultimodalBasicDataset(gen_samples, tx)
        gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size)

        gen_activations = [[],[]]
        for dataT in gen_dataloader:
            data = unpack_data(dataT)
            gen_activations[0].append(model_fid(data[0]))
            gen_activations[1].append(model_fid(data[1]))
        gen_activations = [np.concatenate(g) for g in gen_activations]

        cond_fids = {}
        for i in range(len(ref_activations)):
            mu1, mu2 = np.mean(ref_activations[i], axis=0), np.mean(gen_activations[i], axis=0)
            sigma1, sigma2 = np.cov(ref_activations[i], rowvar=False), np.cov(gen_activations[i], rowvar=False)

            # print(mu1.shape, sigma1.shape)

            cond_fids[f'fid_{i}'] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return cond_fids

            
        




