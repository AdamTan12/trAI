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
        self.feature_extraction = nn.Sequential(
            # 3, 256, 256
            nn.Conv2d(nc, nf, 4, stride = 2, padding=1),

            # nf, 128, 128
            nn.Conv2d(nf, nf*2, 4, stride = 2, padding=1),      nn.BatchNorm2d(nf*2),       nn.ReLU(),
            # nf*2, 64, 64
            nn.Conv2d(nf*2, nf*4, 4, stride = 2, padding=1),    nn.BatchNorm2d(nf*4),       nn.ReLU(),
            # nf*4, 32, 32
            nn.Conv2d(nf*4, nf*8, 4, stride = 2, padding=1),    nn.BatchNorm2d(nf*8),       nn.ReLU(), 
            # nf*8, 16, 16
            nn.Conv2d(nf*8, nf*16, 4, stride = 2, padding=1),   nn.BatchNorm2d(nf*16),       nn.ReLU(),
            # nf*16, 8, 8
        )
        self.feature_interpretation = nn.Sequential(
            nn.Linear(nf*16*64, 1024), nn.BatchNorm1d(1024), nn.ReLU(),

            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),

            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),

        )
        # we do not use a final relu function because we need outputs to be negative as well
        self.bbox_head = nn.Sequential(
            #need this to be 10x4
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 50),
            nn.Sigmoid()
        )
        self.category_head = nn.Sequential(
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10*60)
        )
    def __call__(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.feature_interpretation(x)
        bbox_x = self.bbox_head(x)
        category_x = self.category_head(x)
        bbox_x = bbox_x.view(-1, 10, 5)
        category_x = category_x.view(-1, 10, 60)
        self.bbox_out = bbox_x
        self.category_out = category_x
        return self.bbox_out, self.category_out
    def parameters(self):
        return list(self.feature_extraction.parameters()) + list(self.feature_interpretation.parameters()) + list(self.bbox_head.parameters()) + list(self.category_head.parameters())