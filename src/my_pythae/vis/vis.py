

def plot_embeddings_colorbars(emb0,emb_l0,filepath):
    fig, ax = plt.subplots(1,1)
    sc1 = ax[0].scatter(emb0[:,0],emb0[:,1], c = emb_l0)
    fig.colorbar(sc1,ax=ax[0])
    plt.savefig(filepath)
    plt.close()