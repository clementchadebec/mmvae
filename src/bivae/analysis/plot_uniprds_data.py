''' Compare several models by plotting their prd plots'''

import argparse
import numpy as np
import glob, os
from prd import plot, prd_to_max_f_beta_pair, plot_beta_pairs

parser = argparse.ArgumentParser(description='Compare PRD plots for several models')

parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True, type=str)
parser.add_argument('-n', '--names', nargs='+', type=str)
info = parser.parse_args()

# Load the prd datas for all models
prds = []
for model in info.list:
    # get the latest validate folder
    path = max(glob.glob(os.path.join(model, '*/')), key=os.path.getmtime)
    prds.append(np.load(path + 'uniprd_data_0.npy'))

# Plot
for i,type in enumerate(['Modalité 0', 'Modalité 1']):
    prds_to_plot = []
    f_beta = []
    f_beta_inv = []
    for j, name in enumerate(info.names):
        prds_to_plot.append(prds[j][i])
        precision, recall = prds[j][i]
        fbeta, fbetainv = prd_to_max_f_beta_pair(precision, recall)
        f_beta.append(fbeta)
        f_beta_inv.append(fbetainv)
    plot(prds_to_plot, labels = info.names, out_path='../experiments/comparison/prd_compare_{}.png'.format(type),)
    plot_beta_pairs(f_beta, f_beta_inv, info.names, outpath = '../experiments/comparison/f_beta_pairs_{}.png'.format(type))

