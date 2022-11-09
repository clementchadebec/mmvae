''' Function for CelebA dataset that distributes over the different celeba models'''

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import wandb
from torchvision.utils import save_image

from bivae.dataloaders import BasicDataset
from bivae.utils import unpack_data, add_channels, adjust_shape
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
    preds2 = model.classifiers[1](cross_samples[0].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod2))  # 8*n x 40
    labels2 = (preds2 > 0).int().reshape(n_data, ns,40)

    preds1 = model.classifiers[0](cross_samples[1].permute(1, 0, 2, 3, 4).resize(n_data * ns, *model.shape_mod1))  # 8*n x 10
    labels1 = (preds1 > 0).int().reshape(n_data, ns, 40)
    classes_mul = torch.stack([classes[0][:n_data] for _ in range(ns)]).permute(1, 0,2).cuda()
    # print(classes_mul.shape)

    acc2 = torch.sum(classes_mul == labels2) / (n_data * ns*40)
    acc1 = torch.sum(classes_mul == labels1) / (n_data * ns*40)

    metrics = dict(accuracy1=acc1, accuracy2=acc2)

    # Compute the joint accuracy
    data = model.generate('', 0, N=ns, save=False)
    labels_celeb = model.classifiers[0](data[0]) > 0
    labels_attributes = model.classifiers[1](data[1]) > 0

    joint_acc = torch.sum(labels_attributes == labels_celeb) / (ns * 40)
    metrics['joint_coherence'] = joint_acc

    
    return metrics



def compute_fid_celeba(model, batch_size):

        # Define the inception model used to compute FID
        model_fid = wrapper_inception()

        # Get the data with suited transform
        tx = transforms.Compose([transforms.ToTensor(), transforms.Resize((299, 299)), add_channels()])

        _, test,_ = model.getDataLoaders(batch_size,transform=tx)

        ref_activations = []

        for dataT in test:
            data = unpack_data(dataT)

            ref_activations.append(model_fid(data[0]))

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
            gen_activations.append(model_fid(data[0]))
        gen_activations = np.concatenate(gen_activations)

        # print(ref_activations.shape, gen_activations.shape)

        mu1, mu2 = np.mean(ref_activations, axis=0), np.mean(gen_activations, axis=0)
        sigma1, sigma2 = np.cov(ref_activations, rowvar=False), np.cov(gen_activations, rowvar=False)

        # print(mu1.shape, sigma1.shape)

        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # print(fid)
        return {'fid' : fid}


 

def attribute_array_to_image(tensor, device='cuda'):


    """tensor of size (n_batch, 1,1,40)

    output size (3,64,64)
    """
    list_images=[]
    for v in tensor:
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        fnt =  ImageFont.load_default()#ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 11)
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



def sample_from_conditional_celeba(model, data, runPath, epoch, n=10):
        """Sample from conditional with vector attributes transformed into words"""

        bdata = [d[:8] for d in data]
        model.eval()
        samples = model._sample_from_conditional(bdata, n)

        for r, recon_list in enumerate(samples):
            for o, recon in enumerate(recon_list):
                _data = bdata[r].cpu()
                recon = torch.stack(recon)
                _,_,ch,w,h = recon.shape
                recon = recon.resize(n * 8, ch, w, h).cpu()

                if r == 0 and o == 1:
                    recon = attribute_array_to_image(recon, device='cpu')
                elif r == 1 and o == 0:
                    _data = attribute_array_to_image(_data, device='cpu')

                if _data.shape[1:] != recon.shape[1:]:
                        _data, recon = adjust_shape(_data, recon) # modify the shapes in place to match dimensions

                comp = torch.cat([_data, recon])
                filename = '{}/cond_samples_{}x{}_{:03d}.png'.format(runPath, r, o, epoch)
                save_image(comp, filename)
                wandb.log({'cond_samples_{}x{}.png'.format(r,o) : wandb.Image(filename)})








