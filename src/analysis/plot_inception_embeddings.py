''' Compare several models by plotting their prd plots'''

import argparse
import numpy as np
import glob, os
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from matplotlib.lines import Line2D
# from umap import UMAP
from torchvision.utils import make_grid, save_image

def custom_cmap(n):
    """Create customised colormap for scattered latent plot of n categories.
    Returns colormap object and colormap array that contains the RGB value of the colors.
    See official matplotlib document for colormap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    """
    # first color is grey from Set1, rest other sensible categorical colourmap
    cmap_array = sns.color_palette("Set1", 9)[-1:] + sns.husl_palette(n - 1, h=.6, s=0.7)
    cmap = colors.LinearSegmentedColormap.from_list('mmdgm_cmap', cmap_array)
    return cmap, cmap_array

def plot_embeddings(emb, emb_l, labels, filepath, ticks=None, K=1):

    cmap_obj, cmap_arr = custom_cmap(n=len(labels))
    plt.figure(figsize=(15,10))
    plt.scatter(emb[:, 0], emb[:, 1], c=emb_l, cmap=cmap_obj, s=25, alpha=0.3, edgecolors='none')
    l_elems = [Line2D([0], [0], marker='o', color=cm, label=l, alpha=0.5, linestyle='None')
               for (cm, l) in zip(cmap_arr, labels)]
    plt.legend(frameon=False, loc=2, handles=l_elems)

    # Add some ticks allowing to attach the examples to their latent representation
    m_emb = np.mean(emb.reshape(-1, K, emb.shape[-1]), axis=1)

    if ticks is not None:
        for i, txt in enumerate(ticks[:8]):
            plt.text(m_emb[len(m_emb)//3 + i,0], m_emb[len(m_emb)//3 + i,1], str(txt), c='blue')
            plt.text(m_emb[2*len(m_emb)//3 + i,0], m_emb[2*len(m_emb)//3 + i,1], str(txt), c='red')

    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

parser = argparse.ArgumentParser(description='Compare Inception embeddings between several models and real data')

parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True, type=str)
parser.add_argument('-n', '--names', nargs='+', type=str)
info = parser.parse_args()

# Load the prd datas for all models
activations0 = []
for model in info.list:
    # get the latest validate folder
    path = max(glob.glob(os.path.join(model, '*/')), key=os.path.getmtime)
    activations0.append(np.load(path + 'concat_activ_0.npy'))

print('Activations shape : ', activations0[0].shape)


umap = UMAP(n_neighbors=40)
concat_activations = np.concatenate([activations0[0],activations0[1][len(activations0[1])//2:]])[::-1]
labels = np.concatenate([np.zeros(len(activations0[0])//2), np.ones(len(activations0[0])//2), 2*np.ones(len(activations0[1])//2)])[::-1]

embeddings = umap.fit_transform(concat_activations)
plot_embeddings(embeddings, labels,np.unique(labels), '../experiments/comparison/inception_embeddings.png')


# Do the same for unimodal embeddings
umap = UMAP(n_neighbors=40)
concat_activations = np.concatenate([activations0[0][:,2048:],activations0[1][len(activations0[1])//2:, 2048:]])[::-1]
labels = np.concatenate([np.zeros(len(activations0[0])//2), np.ones(len(activations0[0])//2), 2*np.ones(len(activations0[1])//2)])[::-1]

embeddings = umap.fit_transform(concat_activations)
plot_embeddings(embeddings, labels,np.unique(labels), '../experiments/comparison/inception_embeddings_fmnist.png')

umap = UMAP(n_neighbors=40)
concat_activations = np.concatenate([activations0[0][:,:2048],activations0[1][len(activations0[1])//2:, :2048]])[::-1]
labels = np.concatenate([np.zeros(len(activations0[0])//2), np.ones(len(activations0[0])//2), 2*np.ones(len(activations0[1])//2)])[::-1]

embeddings = umap.fit_transform(concat_activations)
plot_embeddings(embeddings, labels,np.unique(labels), '../experiments/comparison/inception_embeddings_mnist.png')