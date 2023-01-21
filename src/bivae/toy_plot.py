import argparse
import datetime
import json
import random
import sys
from pathlib import Path
import os, glob
from tqdm import tqdm

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

import models
from models.samplers import GaussianMixtureSampler
from utils import Logger, Timer, unpack_data, update_dict_list, get_mean_std, print_mean_std
from torchvision.utils import save_image

'''
Sample from the unimodal posterior for the circles dataset '''

models_names = ['jmvae/circles', 'jmvae_nf/circles', 'jmvae_nf_dcca/circles']
model_pretty_names = ['JMVAE', 'JNF', 'JNF_DCCA']
# models_names = ['jmvae/circles', 'jmvae_nf_recon/circles', 'jmvae_nf_dcca_recon/circles']
models_args = []
models_list = []


wandb.init(project = 'toy_plot' , entity="asenellart") 
wandb.define_metric('epoch')
wandb.define_metric('*', step_metric='epoch')





for name in models_names:
    # load args from disk if pretrained model path is given
    day_path = max(glob.glob(os.path.join('../experiments/' + name, '*/')), key=os.path.getmtime)
    model_path = max(glob.glob(os.path.join(day_path, '*/')), key=os.path.getmtime)
    with open(model_path + 'args.json', 'r') as fcc_file:
        args = argparse.Namespace()
        args.__dict__.update(json.load(fcc_file))
        
    args.device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
    print(f'Device is {args.device}')
    device = torch.device(args.device)

    # Create instance of the model
    print(args.model)
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    model = modelC(args).to(device)
    # Load state_dict from training
    print('Loading model {} from {}'.format(model.modelName, model_path))
    model.load_state_dict(torch.load(model_path + '/model.pt'))
    model._pz_params = model._pz_params
    
    models_list.append(model)
    models_args.append(args)
    

    

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)



plt.tight_layout()


# set up run path
runId = datetime.datetime.now().isoformat()

runPath = Path('toy_compare' + '/validate_'+runId)

runPath.mkdir(parents=True, exist_ok=True)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)


train_loader, test_loader, val_loader = models_list[0].getDataLoaders(500, device=device)
print(f"Train : {len(train_loader.dataset)},"
      f"Test : {len(test_loader.dataset)},"
      f"Val : {len(val_loader.dataset)}")





def eval():
    """Compute all metrics on the entire test dataset"""

    
    b_metrics = {}
    with torch.no_grad():
        for i, dataT in enumerate(tqdm(test_loader)):
            data = unpack_data(dataT, device=device)
            # print(dataT)
            classes = dataT[0][1], dataT[1][1]
            rayons = dataT[0][2], dataT[1][2]
            # Compute the classification accuracies
            if i == 0:
                             
                
                    
                    # try: 
                    #     model.sample_from_poe(data, runPath, 0, n=10, divide_prior=True)
                    # except:
                    #     print('No function implemented for poe generation')
                    
                    # model.reconstruct(data, runPath, epoch=0)
                    # model.analyse(data, runPath, epoch=0, classes=classes)
                    # model.analyse_posterior(data, n_samples=10, runPath=runPath, epoch=0, ticks=None, N=100)
                    # model.generate(runPath, epoch=0, N=32, save=True)
                    # model.generate_from_conditional(runPath, epoch=0, N=32, save=True)
                
                for idx in [1]:
                    plt.rc('font', size=8)          # controls default text sizes
                    plt.rc('axes', titlesize=8)     # fontsize of the axes title
                    plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
                    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
                    plt.rc('legend', fontsize=8)    # legend fontsize
                    
                    fig_width = 487.8225/72.27  * 1.7
                    fig_height = fig_width/3
                    fig_size = (fig_width, fig_height)
                
                    
                    fig1, axs1 = plt.subplots(1,len(models_list), figsize = fig_size, sharex=True, sharey=True)
                    fig2, axs2 = plt.subplots(1, len(models_list), figsize = fig_size, sharex=True, sharey=True)
                                        
                    axs1[0].set_xlim([-4.5,4.5])
                    axs1[0].set_ylim([-4.5,4.5])
                    
                    axs2[0].set_xlim([-4.5,4.5])
                    axs2[0].set_ylim([-4.5,4.5])

                    plots=[(fig1, axs1), (fig2, axs2)]
                    for mod in range(2):
                        for i,model in enumerate(models_list):
                            model.eval()
                            sc = model.plot_joint_and_uni(data, rayons, classes,plots[mod][1][i], plots[mod][0], idx, mod, N=1000)
                            plots[mod][1][i].set_title(model_pretty_names[i])
                            # sample conditional samples from this model
                            model.sample_from_conditional([torch.stack([data[m][idx]]*8) for m in range(2)], runPath,i,n=20)

                        plots[mod][0].tight_layout()
                        plots[mod][0].subplots_adjust(right=0.91)
                        cax = plots[mod][0].add_axes([0.93, 0.06, 0.015, 0.90])
                        t = [-1. , -0.8, -0.6, -0.4, -0.2,  0.  , 0.2 , 0.4 , 0.6,  0.8 , 1. ]
                        cb = plots[mod][0].colorbar(sc, cax=cax, ticks=t)
                        cb.ax.set_yticklabels(np.abs(t))
                        c_label = 'Square size' if mod==0 else 'Circle size'
                        cb.set_label(c_label)
                        # Save the image 
                        save_image(data[mod][idx], str(runPath) + '/image_{}_{}.png'.format(mod, idx))

                    fig1.savefig(str(runPath) + '/joint_uni_mod_1_{}.pdf'.format(idx))    
                    fig2.savefig(str(runPath) + '/joint_uni_mod_2_{}.pdf'.format(idx))
                    plt.close()
                    
                break


    return {}



if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        run_metrics = {}
        for r in range(1): # 5 independants runs to have a std deviation on each value
            eval()
        
        