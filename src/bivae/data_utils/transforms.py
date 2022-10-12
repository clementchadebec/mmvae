import torch
from skimage.feature import canny
from torchvision.transforms import Compose, ToTensor
import numpy as np

class canny_filter(object):
    """
    Apply Canny filter to the images.
    """

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, image):
        cannim = canny(image[0].numpy(), sigma=self.sigma)*1
        cannim= torch.from_numpy(cannim).unsqueeze(0).float()

        return cannim

contour_transform = Compose([ToTensor(),canny_filter()])

class random_grey(object):

    """ Changes the intensity of mnist images"""

    def __call__(self, image):
        intensity_change = np.random.uniform(0.3,1)
        return image*intensity_change

random_grey_transform = Compose([ToTensor(), random_grey()])