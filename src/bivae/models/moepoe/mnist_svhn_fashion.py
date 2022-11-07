"MoePoe specification for MNIST-SVHN-Fashion"

# JMVAE_NF specification for MNIST-SVHN experiment


import torch
import torch.distributions as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
from pythae.models import VAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from torchvision import transforms

from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance
from bivae.dataloaders import MNIST_SVHN_FASHION_DL, BasicDataset
from bivae.my_pythae.models import my_VAE
from bivae.utils import update_details, unpack_data, add_channels
from bivae.vis import plot_hist
from .moepoe import MOEPOE
from ..nn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from bivae.analysis import load_pretrained_svhn, load_pretrained_mnist, compute_accuracies







class MSF(MOEPOE):
    def __init__(self, params):
        vae_config = VAEConfig
        vae = my_VAE

        vae_config1 = vae_config((1,28,28), params.latent_dim)
        vae_config2 = vae_config((3,32,32), params.latent_dim)
        vae_config3 = vae_config((1,28,28), params.latent_dim)


        encoder1, encoder2, encoder3 = Encoder_VAE_MLP(vae_config1), Encoder_VAE_SVHN(vae_config2), Encoder_VAE_MLP(vae_config3) # Standard MLP for
        # encoder1, encoder2 = None, None
        decoder1, decoder2, decoder3 = Decoder_AE_MLP(vae_config1), Decoder_VAE_SVHN(vae_config2), Decoder_AE_MLP(vae_config3)
        # decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2),
            vae(vae_config3, encoder3, decoder3)

        ])
        super(MSF, self).__init__(params, vaes)
        self.modelName = 'moepoe_mnist_svhn_fashion'
        self.data_path = params.data_path
        self.params = params
        self.vaes[0].modelName = 'mnist'
        self.vaes[1].modelName = 'svhn'
        self.vaes[2].modelName = 'fashion'
        self.lik_scaling = ((3 * 32 * 32) / (1 * 28 * 28), 1, (3 * 32 * 32)) if params.llik_scaling == 0 \
            else (params.llik_scaling, 1, params.llik_scaling)


    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = MNIST_SVHN_FASHION_DL(self.data_path).getDataLoaders(batch_size, shuffle, device, transform)
        return train, test, val

    def compute_metrics(self, data, runPath, epoch, classes, n_data=20, ns=100, freq=10):

        """ We want to evaluate how much of the generated samples are actually in the right classes and if
        they are well distributed in that class"""

        # Compute general metrics (FID)
        general_metrics = MOEPOE.compute_metrics(self,runPath,epoch,freq=freq)
        # general_metrics = {}
        # Compute cross_coherence
        labels = self.conditional_labels(data, n_data, ns)
        # Create an extended classes array where each original label is replicated ns times
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1,0).cuda()
        
        accuracies = [[None for _ in range(self.mod)] for _ in range(self.mod)]
        for i in range(self.mod):
            for j in range(self.mod):
                if i!=j:
                    accuracies[i][j] = torch.sum(classes_mul == labels[i][j])/(n_data*ns)
        
        acc_names = [f'acc_{i}_{j}' for i in range(self.mod) for j in range(self.mod) if i!=j]
        acc = [accuracies[i][j] for i in range(self.mod) for j in range(self.mod) if i!=j]
        metrics = dict(zip(acc_names,acc))

        # Compute joint-coherence
        data = self.generate(runPath, epoch, N=ns)
        labels_joint = [torch.argmax(self.classifier[i](data[i]), dim=1) for i in range(self.mod)]
        
        pairs_labels = torch.stack([labels_joint[i] == labels_joint[j] for i in range(self.mod) for j in range(self.mod)])
        joint_acc = torch.sum(torch.all(pairs_labels, dim=0))/ns
        metrics['joint_coherence'] = joint_acc
        update_details(metrics, general_metrics)

        return metrics

    def compute_fid(self, batch_size):
        
        #TODO : Check that this function is working

        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test, _ = self.getDataLoaders(batch_size, transform=tx)

        ref_activations = [[] for i in range(self.mod)]

        for dataT in test:
            data = unpack_data(dataT)
            for i in range(self.mod):
                ref_activations[i].append(model(data[i]))
            

        ref_activations = [np.concatenate(r) for r in ref_activations]

        # Generate data from conditional

        _, test, _ = self.getDataLoaders(batch_size)

        gen_samples = [[[] for j in range(self.mod)] for i in range(self.mod)]
        for dataT in test:
            data = unpack_data(dataT)
            gen = self._sample_from_conditional(data, n=1)
            for i in range(self.mod):
                for j in range(self.mod):
                    gen_samples[i][j].extend(gen[i][j])
            

        gen_samples = [[torch.cat(g).squeeze(0) for g in row] for row in gen_samples]
        print(gen_samples[0].shape)
        tx = transforms.Compose([transforms.Resize((299, 299)), add_channels()])

        gen_activations = [[[] for j in range(self.mod)] for i in range( self.mod)]
        
        for i in range(self.mod):
            for j in range(self.mod):
                if i != j :
                    gen = torch.cat(gen_samples[i][j]).squeeze(0)
                    dataset = BasicDataset(gen,tx)
                    dl = DataLoader(dataset, batch_size)
                    # Compute all the activations
                    for data in dl:
                        gen_activations[i][j].append(model[data])
                


        cond_fids = {}
        
        for i in range(self.mod): # modality sampled
            mu_ref = np.mean(ref_activations[i], axis=0)
            sigma_ref = np.cov(ref_activations[i],rowvar=False )
            for j in range(self.mod): # modality we condition on for sampling
                if i != j:
                    # Compute mean and sigma
                    mu_gen = np.mean(np.concatenate(gen_activations[j][i]))
                    sigma_gen = np.cov(np.concatenate(gen[j][i]), rowvar=False)


                    cond_fids[f'fid_{j}_{i}'] = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

        return cond_fids



    def step(self, epoch):
        pass

    

    def set_classifiers(self):
        self.classifier1,self.classifier2 = load_pretrained_mnist(), load_pretrained_svhn()



