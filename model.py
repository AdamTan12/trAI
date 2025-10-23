import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
#from scipy.signal import convolve

#---parameters---
nc = 3              #number of color channels
nz = 100            #length of latent vectors
nf = 64             #length of feature maps

def add_noise(images, noise_std=0.1):
    return images + noise_std * torch.randn_like(images)

class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        # add relu functions between layers
        self.layers = nn.Sequential(
            # 3, 256, 256
            nn.Conv2d(nc, nf, 4, stride = 2, padding=1),

            # nf, 128, 128
            nn.Conv2d(nf, nf*2, 4, stride = 2, padding=1),      nn.ReLU(), 
            # nf*2, 64, 64
            nn.Conv2d(nf*2, nf*4, 4, stride = 2, padding=1),    nn.ReLU(), 
            # nf*4, 32, 32
            nn.Conv2d(nf*4, nf*8, 4, stride = 2, padding=1),    nn.ReLU(), 
            # nf*8, 16, 16
            nn.Conv2d(nf*8, nf*16, 4, stride = 2, padding=1),   nn.ReLU(), 
            # nf*16, 8, 8
            nn.Conv2d(nf*16, nf*32, 4, stride = 2, padding=1),  nn.ReLU(), 
            # nf*32, 4, 4
            nn.Conv2d(nf*32, 4, 4, stride = 1, padding=0),
        )
        # we do not use a final relu function because we need outputs to be negative as well
        self.head = nn.Linear(4, 4)

    def __call__(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        self.out = x
        return self.out
    def parameters(self):
        return list(self.layers.parameters()) + list(self.head.parameters())