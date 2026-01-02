import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import testModel, add_noise
from torchvision.transforms import functional as TF
from PIL import Image
import json

with open('./training_data/data/annotations.json', 'r') as f:
    data = json.load(f)
categories = data["categories"]

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
m = testModel().to(device)
m.load_state_dict(torch.load("weights/model_weights_final2.pth"))
image = Image.open("training_data/data/batch_2/000015.JPG").convert("RGB")
image_rgb = image.resize((256, 256))
image_tensor = TF.to_tensor(image_rgb)
image_tensor = image_tensor.unsqueeze(0) 
m.eval()
bb_output, cat_output = m(image_tensor.to(device))
bb_output = bb_output.view(bb_output.size(0), -1)
bb_output = bb_output.tolist()
image_np = np.array(image)

cat_idx = cat_output.argmax(dim=-1)

plt.figure(figsize=(6, 6))
plt.imshow(image_np)

width, height = image.size

for i in range(10):
    name = categories[cat_idx[0, i].item()]["name"]
    x = bb_output[0][i*5 + 0] * width
    y = bb_output[0][i*5 + 1] * height
    w = bb_output[0][i*5 + 2] * width
    h = bb_output[0][i*5 + 3] * height
    if (bb_output[0][i*5 + 4] < 0.5):
        continue

    plt.plot([x, x+w], [y, y], color = 'red', linewidth = 2)
    plt.plot([x, x], [y, y+h], color = 'red', linewidth = 2)
    plt.plot([x, x+w], [y+h, y+h], color = 'red', linewidth = 2)
    plt.plot([x+w, x+w], [y, y+h], color = 'red', linewidth = 2)
    plt.text(x, y-5, name, color='red', fontsize=12, backgroundcolor='white')




plt.savefig("bb_output_image.png")