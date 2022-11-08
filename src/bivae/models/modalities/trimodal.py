
from torchvision import transforms
from bivae.utils import unpack_data, add_channels
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as dist

from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.dataloaders import BasicDataset

def fid(model, batch_size):
        
        #TODO : Check that this function is working

        model_fid = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = model.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[] for i in range(model.mod)]

        for dataT in test:
            data = unpack_data(dataT)
            for i in range(model.mod):
                ref_activations[i].append(model_fid(data[i]))
            

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = model.getDataLoaders(batch_size)

        gen_samples = [[[] for j in range(model.mod)] for i in range(model.mod)]
        for dataT in test:
            data = unpack_data(dataT)
            gen = model._sample_from_conditional(data, n=1)
            for i in range(model.mod):
                for j in range(model.mod):
                    gen_samples[i][j].extend(gen[i][j])
            

        gen_samples = [[torch.cat(g).squeeze(0) for g in row] for row in gen_samples]

        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_activations = [[[] for j in range(model.mod)] for i in range( model.mod)]
        
        for i in range(model.mod):
            for j in range(model.mod):
                if i != j :
                    gen = gen_samples[i][j]
                    dataset = BasicDataset(gen,tx)
                    dl = DataLoader(dataset, batch_size)
                    # Compute all the activations
                    for data in dl:
                        gen_activations[i][j].append(model_fid(data[0]))
                


        cond_fids = {}
        
        for i in range(model.mod): # modality sampled
            mu_ref = np.mean(ref_activations[i], axis=0)
            sigma_ref = np.cov(ref_activations[i],rowvar=False )
            for j in range(model.mod): # modality we condition on for sampling
                if i != j:
                    # Compute mean and sigma
                    mu_gen = np.mean(np.concatenate(gen_activations[j][i]), axis=0)
                    sigma_gen = np.cov(np.concatenate(gen_activations[j][i]), rowvar=False)

                    cond_fids[f'fid_{j}_{i}'] = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

        return cond_fids
    

def sample_from_conditional_subset(self, subsets, data, ns):
        
        """
        We define the distribution p(z|s) where s is a subset of modality as a mixture of experts. 
        Here we sample from this conditional.
        
        Parameters
        ----------
        
        subset : List of lists
            The subsets we condition on. for each subset, the first indices correpond to the one we condition on and the last one
            correspond to the modality sampled.
            
        ns : int
            The number of samples to produce
            
        Returns 
        -------------
        
        recons : List of Tensors
            Contains the generated samples for each subset 
        """
        
        uni_modals = self._sample_from_conditional(data,ns)
        recons = []
        for subset in subsets:
            s = dist.Categorical(torch.ones(len(subset-1))/len(subset-1)).sample()
            
            recons.append(uni_modals[subset[s]][subset[-1]])
        return recons
    








