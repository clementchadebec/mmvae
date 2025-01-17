
from torchvision import transforms
from bivae.utils import unpack_data, add_channels
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as dist
import wandb
from torchvision.utils import save_image

from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.dataloaders import BasicDataset
import time

def fid(model, batch_size):
        
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
    

def sample_from_moe_subset(model, subsets, data, ns):
        
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
        
        uni_modals = model._sample_from_conditional(data,ns)
        recons = []
        for subset in subsets:
            s = dist.Categorical(torch.ones(len(subset-1))/len(subset-1)).sample()
            
            recons.append(uni_modals[subset[s]][subset[-1]])
        return recons
    



def compute_poe_subset_accuracy(model,data, classes,n_data=100,ns=100):
    """Compute the accuracy when generating from a subset of modality, using the poe subsampling.
    
    Args:
        data (list): batch
        subset (list): the modalities to condition on
        gen_mod (int) : the modality to generate in
        classes (tensor): the labels
        n_data (int, optional): number of datapoints to consider. Defaults to 100.
        ns (int, optional): The number of samples per datapoint. Defaults to 100.
    """

    if n_data == 'all' or n_data > len(data[0]):
        n_data = len(data[0])
    
    subsets = [[1,2],[0,2],[0,1]]
    bdata = [d[:n_data] for d in data]
    mult_true_classes = torch.cat([classes[0][:n_data]]*ns) #(ns*n_data)
    
    r_dict = {}
    
    for s,gen_mod in zip(subsets,range(3)):
        
        # Sample from the conditional subset
        zs = model.sample_from_poe_subset(s, bdata, K = ns, divide_prior=True) # ns x n_data x latent_dim
        
        # Reconstruct in the last modality
        with torch.no_grad():
            r = model.vaes[gen_mod].decoder(zs.reshape(ns*n_data, -1)).reconstruction # (ns*n_data , ch,w,h)
        
        # Classify the generated data
        preds = model.classifiers[gen_mod](r) #probs
        preds = torch.argmax(preds, dim=1).cpu() #categorical labels
        
        # Compare with true classes
        acc= torch.sum(preds == mult_true_classes)/len(preds)
        
        r_dict['cond_acc_{}'.format(gen_mod)] = acc

    return r_dict



def compute_cond_ll_from_poe_subset(model, data, subset, gen_mod, K=1000, batch_size_K=200,i_s=0):
    """
    Compute the likehoods of the generation from conditional poe subset posteriors.
    Function used with MVAE and JMVAE. 
    We use one of the two unimodal distributions as the importance distribution.

    Args:
        model (MVAE or JMVAE_NF instance)
        data (list): _description_
        subset (list): _description_
        gen_mod (int): _description_
        K (int, optional): number of samples to estimate likelihoods. Defaults to 1000.
        i_s (int) : the distribution to use for importance sampling (0 for the first of the subset,1 for the second)

    Returns:
        float: average likelihoods across the datapoints
    """
    

    # Then iter on each datapoint to compute the iwae estimate of ln(p(x|y))
    ll = []
    nb_batches = K // batch_size_K
    for i in range(len(data[0])):
        # Choose at random a modality to use for the importance sampling
        i_s = np.random.randint(0,2)
        i_s_mod, n_i_s_mod = subset[i_s], subset[1-i_s]
        normalize_weights = []
        for n in range(nb_batches):
            lnpxs = []
            repeated_data_point = torch.stack(batch_size_K * [data[i_s_mod][i]]) # batch_size_K, n_channels, h, w
            repeated_data_subset = [torch.stack(batch_size_K * [data[m][i]]) for m in range(model.mod)]        

            # Encode with the conditional VAE
            latents = model.infer_latent_from_mod(i_s_mod,repeated_data_point)
            # Decode with the opposite decoder
            recon = model.vaes[gen_mod].decoder(latents).reconstruction

            # Compute lnp(y|z)


            if model.px_z[gen_mod] == dist.Bernoulli:
                lpx_z = model.px_z[gen_mod](recon).log_prob(data[gen_mod][i]).sum(dim=(1, 2, 3))
            else:
                lpx_z = model.px_z[gen_mod](recon, scale=1).log_prob(data[gen_mod][i]).sum(dim=(1, 2, 3))

            # Compute Importance weights 
            # As subset we only consider the modality we didn't use to sample from since w = q(z|x_1)q(z|x_2)/(p(z)*q(z|x_2)) if i_s=2
            ln_q_zxs = model.compute_poe_posterior([n_i_s_mod],latents,repeated_data_subset, divide_prior=True, grad=False)

            
            lpx_z += ln_q_zxs
            normalize_weights.append(torch.logsumexp(ln_q_zxs, dim=0))
            lnpxs.append(torch.logsumexp(torch.Tensor(lpx_z), dim=0) ) 
        
        normalize_weights = torch.logsumexp(torch.Tensor(normalize_weights), dim=0)
        ll.append(torch.logsumexp(torch.Tensor(lnpxs), dim=0) - normalize_weights)

    return torch.sum(torch.tensor(ll))/len(ll)





def compute_all_cond_ll_from_poe_subsets(model,data, K=1000,batch_size_K=100):
    subsets = [[1,2],[0,2],[0,1]]
    r_dict = {}
    
    for s,gen_mod in zip(subsets, range(3)):
        
        ll = compute_cond_ll_from_poe_subset(model,data,s,gen_mod,K)
        r_dict['cond_poe_ll_{}'.format(gen_mod)] = ll
    return r_dict


def sample_from_poe_vis(model, data, runPath, epoch, n=10,divide_prior=False):
        """ Visualize the conditional distribution using the poe subset

        Args:
            data (list): _description_
            runPath (str): _description_
            epoch (int): _description_
            n (int, optional): number of samples per datapoint. Defaults to 10.
            
        """
        print('passing poe vis', divide_prior)
        
        b_data = [d[:8] for d in data]
        subsets = [[1,2],[0,2],[0,1]]
        for s,gen_mod in zip(subsets, range(3)):
            # Sample
            zs = model.sample_from_poe_subset(s,b_data, K=n, divide_prior=divide_prior) # n x 8 x latent_dim
            # Reconstruct
            with torch.no_grad():
                r = model.vaes[gen_mod].decoder(zs.reshape(n*len(b_data[0]), -1)).reconstruction 
            
            filename = '{}/cond_samples_subset{}_{}.png'.format(runPath,str(s), gen_mod)
            save_image(torch.cat([b_data[gen_mod],r]), filename)
            wandb.log({'cond_samples_subset{}.png'.format(gen_mod) : wandb.Image(filename)})

    