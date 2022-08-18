''' Compare several models by plotting their prd plots'''

import argparse
import numpy as np
import glob, os
from prd import plot

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
    for j, name in enumerate(info.names):
        prds_to_plot.append(prds[j][i])
    plot(prds_to_plot, labels = info.names, out_path='../experiments/comparison/prd_compare_{}.png'.format(type),)

