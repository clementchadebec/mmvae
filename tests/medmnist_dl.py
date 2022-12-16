''' Visualize samples from mnist-contour dataset'''

from bivae.dataloaders import PATH_BLOOD_DL
from bivae.utils import *
import torch
from torchvision.utils import save_image

train, test, val = PATH_BLOOD_DL().getDataLoaders(batch_size=10)
print(len(train.dataset), len(test.dataset), len(val.dataset))
print(train.batch_size)
# print(len(train.dataset))
samples = next(iter(train))
print(samples[0][1] == samples[1][1])

data = unpack_data(samples)
print(data[0].shape, data[1].shape)
save_image(data[1],'./tests/mnist_contour.png')
