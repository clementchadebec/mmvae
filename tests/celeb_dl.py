
from bivae.dataloaders import CELEBA_DL
from torchvision.utils import save_image
from bivae.utils import unpack_data
import torch

train, test, val = CELEBA_DL().getDataLoaders(10)

samples = next(iter(train))
data = unpack_data(samples)

print(data[1][0])

save_image(data[0], 'tests/celeb_dl.png')

