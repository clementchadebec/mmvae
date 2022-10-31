# MMVAE specification for CelebA experiment

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.manifold import TSNE
import wandb
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

from bivae.utils import get_mean, kl_divergence, negative_entropy, add_channels, update_details
from bivae.vis import tensors_to_df, plot_embeddings_colorbars, plot_samples_posteriors, plot_hist, save_samples_mnist_svhn

from pythae.models import VAE_LinNF_Config, VAE_IAF_Config, VAEConfig
from bivae.my_pythae.models import my_VAE_LinNF, my_VAE_IAF, my_VAE, laplace_VAE
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA

from bivae.dataloaders import CELEBA_DL, BasicDataset
from ..nn import Encoder_VAE_MNIST, Decoder_AE_MNIST, Encoder_VAE_SVHN, Decoder_VAE_SVHN
from bivae.analysis.classifiers.CelebA_classifier import load_celeba_classifiers
from bivae.utils import adjust_shape, unpack_data
from bivae.vis import save_samples
from torch.utils.data import DataLoader

from ..nn import DoubleHeadMLP, DoubleHeadJoint
from .mmvae import MMVAE
from bivae.analysis.pytorch_fid import wrapper_inception, calculate_frechet_distance







class celeba(MMVAE):
    def __init__(self, params):
        vae_config = VAEConfig

        self.shape_mod1 = (3,64,64)
        self.shape_mod2 = (1,1,40)

        vae_config1 = vae_config((3,64,64), params.latent_dim)
        vae_config2 = vae_config((1,1,40), params.latent_dim)
        vae = my_VAE if params.dist == 'normal' else laplace_VAE

        encoder1, encoder2 = Encoder_ResNet_VAE_CELEBA(vae_config1), Encoder_VAE_MLP(vae_config2) # Standard MLP for
        # encoder1, encoder2 = None, None
        decoder1, decoder2 = Decoder_ResNet_AE_CELEBA(vae_config1), Decoder_AE_MLP(vae_config2)
        # decoder1, decoder2 = None, None

        vaes = nn.ModuleList([
            vae(model_config=vae_config1, encoder=encoder1, decoder=decoder1),
            vae(model_config=vae_config2, encoder=encoder2, decoder=decoder2)

        ])
        super(celeba, self).__init__(params, vaes)
        self.modelName = 'mmvae_celeba'

        self.vaes[0].modelName = 'celeb'
        self.vaes[1].modelName = 'attributes'
        self.lik_scaling = (np.prod(self.shape_mod2) / np.prod(self.shape_mod1),1) if params.llik_scaling == 0 else (params.llik_scaling, 1)

        wandb.config.update({'lik_scalings' : self.lik_scaling})

    def getDataLoaders(self, batch_size, shuffle=True, device="cuda", transform = transforms.ToTensor()):
        train, test, val = CELEBA_DL(self.data_path).getDataLoaders(batch_size, shuffle, device)
        return train, test, val





    def compute_metrics(self, data, runPath, epoch, classes, n_data=100, ns=300, freq=10):
        """

        inputs :

        - classes of shape (batch_size, 40)"""



        bdata = [d[:n_data] for d in data]
        samples = self._sample_from_conditional(bdata, n=ns)
        cross_samples = [torch.stack(samples[0][1]), torch.stack(samples[1][0])]

        # Compute the labels
        preds2 = self.classifier2(cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, *self.shape_mod2))  # 8*n x 40
        labels2 = (preds2 > 0).int().reshape(n_data, ns,40)

        preds1 = self.classifier1(cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, *self.shape_mod1))  # 8*n x 10
        labels1 = (preds1 > 0).int().reshape(n_data, ns, 40)
        classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1, 0,2).cuda()
        # print(classes_mul.shape)

        acc2 = torch.sum(classes_mul == labels2) / (n_data * ns*40)
        acc1 = torch.sum(classes_mul == labels1) / (n_data * ns*40)

        metrics = dict(accuracy1=acc1, accuracy2=acc2)

        # Compute the joint accuracy
        data = self.generate('', 0, N=ns, save=False)
        labels_celeb = self.classifier1(data[0]) > 0
        labels_attributes = self.classifier2(data[1]) > 0

        joint_acc = torch.sum(labels_attributes == labels_celeb) / (ns * 40)
        metrics['joint_coherence'] = joint_acc

        general_metrics = MMVAE.compute_metrics(self, runPath, epoch, freq=freq)

        update_details(metrics, general_metrics)
        return metrics


    def attribute_array_to_image(self, tensor, device='cuda'):
        """tensor of size (n_batch, 1,1,40)

        output size (3,64,64)
        """
        list_images=[]
        for v in tensor:
            img = Image.new('RGB', (100, 100), color=(0, 0, 0))
            d = ImageDraw.Draw(img)
            #fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 11)
            fnt = ImageFont.load_default()
            vector = v.squeeze()



            text = "Bald {:.1f} \n"\
                   "Bangs {:.1f}\n"\
                   "Big_Nose {:.1f} \n"\
                   "Blond_Hair {:.1f}\n"\
                   "Eyeglasses {:.1f}\n"\
                   "Male {:.1f}\n"\
                   "No_Beard {:.1f}\n".format(vector[4], vector[5], vector[7], vector[9], vector[15], vector[20], vector[24])

            offset = fnt.getbbox(text)
            d.multiline_text((0 - offset[0], 0 - offset[1]), text, font=fnt)

            list_images.append(torch.from_numpy(np.array(img).transpose([2,0,1])))

        return torch.stack(list_images).to(device) # nb_batch x 3 x 100 x 100

    def generate(self,runPath, epoch, N= 8, save=False):
        """Generate samples from sampling the prior distribution"""
        self.eval()
        with torch.no_grad():
            data = []
            if self.sampler is None:
                pz = self.pz(*self.pz_params)
                latents = pz.rsample(torch.Size([N])).squeeze()
            else :
                latents = self.sampler.sample(num_samples=N)
            for d, vae in enumerate(self.vaes):
                data.append(vae.decoder(latents)["reconstruction"])

        if save:
            data = [*adjust_shape(data[0],self.attribute_array_to_image(data[1]))]
            file = ('{}/generate_{:03d}'+self.save_format).format(runPath, epoch)
            save_samples(data,file)
            wandb.log({'generate_joint' : wandb.Image(file)})
        return data  # list of generations---one for each modality

    def sample_from_conditional(self, data, runPath, epoch, n=10):
        """Sample from conditional with vector attributes transformed into words"""

        bdata = [d[:8] for d in data]
        self.eval()
        samples = self._sample_from_conditional(bdata, n)

        for r, recon_list in enumerate(samples):
            for o, recon in enumerate(recon_list):
                _data = bdata[r].cpu()
                recon = torch.stack(recon)
                _,_,ch,w,h = recon.shape
                recon = recon.resize(n * 8, ch, w, h).cpu()

                if r == 0 and o == 1:
                    recon = self.attribute_array_to_image(recon, device='cpu')
                elif r == 1 and o == 0:
                    _data = self.attribute_array_to_image(_data, device='cpu')

                if _data.shape[1:] != recon.shape[1:]:
                        _data, recon = adjust_shape(_data, recon) # modify the shapes in place to match dimensions

                comp = torch.cat([_data, recon])
                filename = '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch)
                save_image(comp, filename)
                wandb.log({'cond_samples_{}x{}.png'.format(r,o) : wandb.Image(filename)})



    def analyse(self, data, runPath, epoch=0, classes=None):
        return


    def compute_fid(self, batch_size):

        # Define the inception model used to compute FID
        model = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test,_ = self.getDataLoaders(batch_size,transform=tx)

        ref_activations = []

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations.append(model(data[0]))

        ref_activations = np.concatenate(ref_activations)

        # Generate data from conditional

        _, test,_ = self.getDataLoaders(batch_size)

        gen_samples = []
        for dataT in test:
            data=unpack_data(dataT)
            gen = self._sample_from_conditional(data, n=1)[1][0]


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





    def set_classifiers(self):

        # Define the classifiers for analysis
        self.classifier1, self.classifier2 = load_celeba_classifiers()    










