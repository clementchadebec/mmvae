# visualisation related functions

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


def embed_umap(data):
    """data should be on cpu, numpy"""
    embedding = UMAP(metric='euclidean',
                     n_neighbors=40,
                     # angular_rp_forest=True,
                     # random_state=torch.initial_seed(),
                     transform_seed=torch.initial_seed())
    return embedding.fit_transform(data)


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

def plot_embeddings_colorbars(emb0,emb1,emb_l0,emb_l1,filepath, filters=None, ax_lim = [-4,4]):
    point_shapes = ['o', '^']
    if filters is None:
        filters = [torch.ones_like(emb_l0)]
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    if ax_lim is not None:
        ax[0].set_xlim(ax_lim)
        ax[0].set_ylim(ax_lim)
    for m,filter in enumerate(filters):
        filter = filter.numpy().astype(bool)
        emb0f, emb1f = emb0[filter], emb1[filter]
        emb_l0f, emb_l1f = emb_l0[filter], emb_l1[filter]
        sc1 = ax[0].scatter(emb0f[:,0],emb0f[:,1],marker=point_shapes[m], c = emb_l0f)
        sc2 = ax[1].scatter(emb1f[:,0],emb1f[:,1],marker = point_shapes[m], c = emb_l1f)
    fig.colorbar(sc2, ax=ax[1])
    fig.colorbar(sc1, ax=ax[0])
    ax[0].set_xlabel(r'$z_1$')
    ax[1].set_xlabel(r'$z_1$')
    ax[0].set_ylabel(r'$z_2$')
    ax[0].set_title('Rayons des carrés')
    ax[1].set_title('Rayons des cercles')
    plt.suptitle(r'Représentation latente $q_{\phi}(z|x,y)$')
    plt.savefig(filepath)
    plt.close()
    
def plot_joint_latent_space(emb,rayons,ax, fig, filters=None):
    point_shapes = ['o', '^']
    if filters is None:
        filters = [torch.ones_like(rayons)]
    
    for m,filter in enumerate(filters):
        filter = filter.numpy().astype(bool)
        embf = emb[filter]
        rf = rayons[filter]
        sc = ax.scatter(embf[:,0],embf[:,1],marker=point_shapes[m], c = rf)

    fig.colorbar(sc, ax=ax)
    

def plot_posteriors_ellipsoid(means, stds,filepath,labels, ticks = None, colors = ['blue', 'orange', 'green']):
    fig, ax = plt.subplots()
    min = np.min([(torch.min(means[i])-torch.max(stds[i])).cpu() for i in np.arange(len(means))])
    max = np.max([(torch.max(means[i]) + torch.max(stds[i])).cpu() for i in np.arange(len(means))])
    ax.set_xlim(min,max)
    ax.set_ylim(min,max)

    for i,t_means in enumerate(means) : # enumerate on modalities
        t_stds, c = stds[i], colors[i]
        for j,m in enumerate(t_means): # enumerate on samples
            ax.add_patch(Ellipse((m[0],m[1]), t_stds[j][0], t_stds[j][1],figure=fig, color=c, fill=False))
            if ticks is not None :
                ax.text(m[0], m[1], str(ticks[j]))


    l_elems = [Line2D([0], [0], marker='o', color=cm, label=l, linestyle='None')
               for (cm, l) in zip(colors, labels)]
    plt.legend(frameon=False, loc=2, handles=l_elems)
    plt.savefig(filepath, bbox_inches='tight')
    return

def plot_posteriors(means, std, filepath, labels, ticks=None, colors=['blue', 'orange','green']):
    plot_posteriors_ellipsoid(means, std,filepath, labels, ticks, colors)

def plot_samples_posteriors(zsamples, filepath, labels, ticks = None):
    """
    zsamples is a list with zsamples[m] of shape N, n_data_points, laten_dim
    """
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4,4])
    ax[0].set_xlabel(r'$z_1$')
    ax[1].set_xlabel(r'$z_1$')
    ax[0].set_ylabel(r'$z_2$')
    ax[1].set_ylabel(r'$z_2$')

    for m, zs in enumerate(zsamples):
        zs = zs.permute(1,0,2)
        for i, z in enumerate(zs):
            ax[m].scatter(z[:,0].cpu(), z[:,1].cpu(), label=i)
    ax[0].set_title(r'Samples from $q_{\phi_1}(z|x)$ for several $x$')
    ax[1].set_title(r'Samples from $q_{\phi_2}(z|y)$ for several $y$')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')



def tensor_to_df(tensor, ax_names=None):
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df


def plot_kls_df(df, filepath):
    _, cmap_arr = custom_cmap(df[df.columns[0]].nunique() + 1)
    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.FacetGrid(df, height=12, aspect=2)
        g = g.map(sns.boxplot, df.columns[1], df.columns[2], df.columns[0], palette=cmap_arr[1:],
                  order=None, hue_order=None)
        g = g.set(yscale='log').despine(offset=10)
        plt.legend(loc='best', fontsize='22')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

def plot_hist(rayons, filename, range=(0,1), bins=10):
    """rayons tensor (n_data,n_samples)"""

    fig, ax = plt.subplots(rayons.shape[0]//8+rayons.shape[0]%8,min(rayons.shape[0], 8))
    for i, t in enumerate(rayons):
        try :
            ax[i//8, i%8].hist(t.cpu(), bins=bins, range=range, density=True)
        except :
            try :
                ax[i%8].hist(t.cpu(), bins=bins, range=range,density=True)
            except:
                ax.hist(t.cpu(), bins=bins, range=range,density=True)
    plt.suptitle("Histogrammes pour 8 examples q(z|x),q(z|y)")
    plt.savefig(filename)
    plt.close()

def save_samples(data, filename):
    """data is a list containing samples from each modality"""

    grids = torch.cat([make_grid(d) for d in data], dim=2)
    save_image(grids, filename)


def save_samples_mnist_svhn(data, filename):

    mnist = torch.cat([data[0] for _ in range(3)], dim=1)
    save_samples([mnist, data[1]], filename)
