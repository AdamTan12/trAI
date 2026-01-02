import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import testModel, add_noise
from dataset import bbox_dataset
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split
from PIL import Image
import json


#variables
manual_seed = 999
workers = 12
batch_size = 128
image_size = 256
ngpu = 1
num_epochs = 50
#dataset
with open('./training_data/data/annotations.json', 'r') as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

bbox_dataset = bbox_dataset(annotations, images, categories)

#=====================================================================================================
dataset_size = len(bbox_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(bbox_dataset, [train_size, val_size])
#=====================================================================================================
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



m = testModel()
#m.load_state_dict(torch.load("model_weights_final.pth"))
m = m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
train_bb_loss_list = []
train_cat_loss_list = []
train_total_loss_list = []
val_bb_loss_list = []
val_cat_loss_list = []
val_total_loss_list = []
loss_fn = torch.nn.CrossEntropyLoss()
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            m.train()
            optimizer.zero_grad()
            # images: (128, 3, 256, 256)
            # labels: (128, 59) (there are 59 different categories)
            images, labels, cat = data
            images, labels, cat = images.to(device), labels.to(device), cat.to(device) #, mask.to(device)
            # the last batch might not be full size of 128, we can skip it as the model expects batches of 128
            if images.size(0) < batch_size:
                continue
            
            images = add_noise(images)
            # train on data
            bb_output, cat_output = m(images)

            cat_indices = cat.argmax(dim=-1)         # (B, 10)
            padding_mask = cat.sum(dim=-1) == 0      # padded rows
            cat_indices[padding_mask] = -100         # mark ignore rows

            cat_loss = F.cross_entropy(
                cat_output.view(-1, 60),
                cat_indices.view(-1),
                ignore_index=-100
            )
            shared_params = list(m.feature_extraction.parameters()) + list(m.feature_interpretation.parameters())
      
            bb_loss = F.smooth_l1_loss(bb_output, labels)
            optimizer.zero_grad()
            bb_loss.backward(retain_graph=True)
            bbox_grad_norm = sum(p.grad.norm() for p in shared_params)

            optimizer.zero_grad()
            cat_loss.backward(retain_graph=True)
            cat_grad_norm = sum(p.grad.norm() for p in shared_params)

            lambda_cat = bbox_grad_norm / (cat_grad_norm + 1e-8)
            loss_total = bb_loss + lambda_cat * cat_loss

            optimizer.zero_grad()
            #loss_total = 0.025 * cat_loss + bb_loss
            print(str(loss_total.item()) + " = cat: " + str(cat_loss.item()) + " + bbox: " + str(bb_loss.item()))
            loss_total.backward()

            optimizer.step()
        if epoch%10 == 0:
            torch.save(m.state_dict(), "model_weights_" + str((epoch/10)) + ".pth", _use_new_zipfile_serialization=True)

        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                m.eval()
                # images: (128, 3, 256, 256)
                # labels: (128, 59) (there are 59 different categories)
                images, labels, cat = data
                images, labels, cat = images.to(device), labels.to(device), cat.to(device) #, mask.to(device)
                cat_indices = cat.argmax(dim=-1)
                # the last batch might not be full size of 128, we can skip it as the model expects batches of 128
                if images.size(0) < batch_size:
                    continue
                bb_output, cat_output = m(images)
                #masked_output = output * mask
                v_bb_loss = F.smooth_l1_loss(bb_output, labels)
                mask = cat.sum(dim=-1) != 0  # True for real objects
                v_cat_loss = F.cross_entropy(cat_output[mask], cat_indices[mask])
                #v_cat_loss = loss_fn(cat_output, cat)
                v_loss_total = lambda_cat * v_cat_loss + v_bb_loss
            

        train_cat_loss_list.append(lambda_cat.item() * cat_loss.item())
        train_bb_loss_list.append(bb_loss.item())
        train_total_loss_list.append(loss_total.item())

        val_cat_loss_list.append(lambda_cat.item() * v_cat_loss.item())
        val_bb_loss_list.append(v_bb_loss.item())
        val_total_loss_list.append(v_loss_total.item())

image = Image.open("training_data/data/batch_1/000000.jpg").convert("RGB")
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


plt.savefig("output_image.png")
#print(output.shape)

plt.figure(figsize=(8,5))
plt.plot(train_bb_loss_list, label='Bounding Box Training Loss', color = 'red')
plt.plot(val_bb_loss_list, label='Bounding Box Validation Loss', color = 'blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("bb_loss_graph.png")

plt.figure(figsize=(8,5))
plt.plot(train_cat_loss_list, label='Category Training Loss', color = 'red')
plt.plot(val_cat_loss_list, label='Category Validation Loss', color = 'blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("cat_loss_graph.png")

plt.figure(figsize=(8,5))
plt.plot(train_total_loss_list, label='Total Training Loss', color = 'red')
plt.plot(val_total_loss_list, label='Total Validation Loss', color = 'blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("total_loss_graph.png")


torch.save(m.state_dict(), "model_weights_final2.pth", _use_new_zipfile_serialization=True)