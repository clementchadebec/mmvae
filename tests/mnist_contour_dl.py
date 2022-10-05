''' Visualize samples from mnist-contour dataset'''

from bivae.dataloaders import MNIST_CONTOUR_DL
from bivae.utils import *
import torch
from torchvision.utils import save_image

train, test, val = MNIST_CONTOUR_DL().getDataLoaders(10)

samples = next(iter(train))

data = unpack_data(samples)
save_image(torch.cat(data),'/mnist_contour.png')