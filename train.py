import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import model


m = model()

for epoch in range(100):
    for i, data in enumerate(dataloader, 0):
        #trainhere....