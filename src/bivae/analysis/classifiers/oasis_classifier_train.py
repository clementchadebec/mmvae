'''

Define an OASIS classifier to quantify coherence of the joint/conditional generation of multimodal VAEs

'''


import torchvision.models as models
from torch import nn

def create_model():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, 2)
    return resnet18