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
from scipy.signal import convolve

#---parameters---
nc = 3              #number of color channels
nz = 100            #length of latent vectors
nf = 64             #length of feature maps



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layers = [

            # 3, 256, 256
            nn.Conv2d(nc, nf, 4, stride = 2, padding=1),
            # nf, 128, 128
            nn.Conv2d(nf, nf*2, 4, stride = 2, padding=1),
            # nf*2, 64, 64
            nn.Conv2d(nf*2, nf*4, 4, stride = 2, padding=1),
            # nf*4, 32, 32
            nn.Conv2d(nf*4, nf*8, 4, stride = 2, padding=1),
            # nf*8, 16, 16
            nn.Conv2d(nf*8, nf*16, 4, stride = 2, padding=1),
            # nf*16, 8, 8
            nn.Conv2d(nf*16, nf*32, 4, stride = 2, padding=1),
            # nf*32, 4, 4
            nn.Conv2d(nf*32, 4, 4, stride = 1, padding=0),
            # 4, 1, 1
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, nn.Module):  # If it's a valid module (e.g., ConvolutionalLayer)
                params += list(layer.parameters())  # Get parameters of the layer
        return params