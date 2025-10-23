import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import model, add_noise
from dataset import bbox_dataset, category_dataset

import json


#variables
manual_seed = 999
workers = 2
batch_size = 128
image_size = 256
ngpu = 1
num_epochs = 2
#dataset
with open('./training_data/data/annotations.json', 'r') as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

bbox_dataset = bbox_dataset(annotations, images)
category_dataset = category_dataset(annotations, images)

dataloader = torch.utils.data.DataLoader(bbox_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


m = model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad
        # images: (128, 3, 256, 256)
        # labels: (128, 59) (there are 59 different categories)
        images, labels = data

        # the last batch might not be full size of 128, we can skip it as the model expects batches of 128
        if images.size(0) < batch_size:
            continue
        
        images = add_noise(images)
        # train on data
        output = output(images)
        loss = F.smooth_l1_loss(output, labels)

        loss.backward()
        optimizer.step()



