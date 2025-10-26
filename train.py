import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import testModel, add_noise
from dataset import bbox_dataset, category_dataset
from torchvision.transforms import functional as TF
from PIL import Image
import json


#variables
manual_seed = 999
workers = 12
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
category_dataset = category_dataset(annotations, categories, images)

dataloader = torch.utils.data.DataLoader(bbox_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


m = testModel()
m.load_state_dict(torch.load("./checkpoints/model_weights.pth"))
m = m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.0001, betas=(0.5, 0.999))
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            # images: (128, 3, 256, 256)
            # labels: (128, 59) (there are 59 different categories)
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # the last batch might not be full size of 128, we can skip it as the model expects batches of 128
            if images.size(0) < batch_size:
                continue
            
            #images = add_noise(images)
            # train on data
            output = m(images)
            loss = F.smooth_l1_loss(output, labels)
            print(loss.shape)
            print(loss.item())

            loss.backward()
            optimizer.step()


image = Image.open("training_data/data/batch_1/000000.jpg").convert("RGB")
image_rgb = image.resize((256, 256))
image_tensor = TF.to_tensor(image_rgb)
image_tensor = image_tensor.unsqueeze(0) 
output = m(image_tensor.to(device))
output = output.view(output.size(0), -1)
output = output.tolist()
image_np = np.array(image)


plt.figure(figsize=(6, 6))
plt.imshow(image_np)

width, height = image.size

x = output[0][0] * width
y = output[0][1] * height
w = output[0][2] * width
h = output[0][3] * height


plt.plot([x, x+w], [y, y], color = 'red', linewidth = 2)
plt.plot([x, x], [y, y+h], color = 'red', linewidth = 2)
plt.plot([x, x+w], [y+h, y+h], color = 'red', linewidth = 2)
plt.plot([x+w, x+w], [y, y+h], color = 'red', linewidth = 2)


plt.savefig("output_image.png")
#print(output.shape)


torch.save(m.state_dict(), "./checkpoints/model_weights.pth", _use_new_zipfile_serialization=True)


