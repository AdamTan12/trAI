import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torchvision.transforms import functional as TF
from PIL import Image


class bbox_dataset(Dataset):
    """
    creates a Dataset of images and uses four point bounding boxes as labels

    Attributes:
        img_dir (list): List of image paths
        img_labels (list): List of bounding box labels
    """
    def __init__(self, annotations, images, categories):
        self.img_labels = []
        self.img_dir = []
        self.img_cat = []
        # Initialize lists
        # make sure no duplicate images
        for i, annotation in enumerate(annotations):
            if (i != 0 and annotation["image_id"] == annotations[i - 1]["image_id"]):
                continue
            else:
                self.img_dir.append(images[annotation["image_id"]]["file_name"])

        # Append all bbox's to label lists, but make sure each one is scaled to 256x256 pixels

        for i, annotation in enumerate(annotations):
            # scale_x = 256 / images[annotation["image_id"]]["width"]
            # scale_y = 256 / images[annotation["image_id"]]["height"]
            image_width = images[annotation["image_id"]]["width"]
            image_height = images[annotation["image_id"]]["height"]
            x, y, w, h = annotation["bbox"]
            bbox_scaled = [
                x / image_width,
                y / image_height,
                w / image_width,
                h / image_height,
                1 # to mark that there is an object here
            ]
            label = F.one_hot(torch.tensor(annotation["category_id"], dtype=torch.long), num_classes=len(categories)).float().unsqueeze(0)
            label_list = label.squeeze(0).tolist() 
            if (i != 0 and annotation["image_id"] == annotations[i - 1]["image_id"]):
                # add bounding box to previous element in img_labels (because an image can have multiple bounding boxes)
                #self.img_cat[-1] = torch.cat([self.img_cat[-1], label], dim = 0)
                self.img_cat[-1].extend(label_list)
                self.img_labels[-1].extend(bbox_scaled)
            else:
                self.img_labels.append(bbox_scaled)
                self.img_cat.append(label_list)
        # for i, annotation in enumerate(annotations):
        #     #label = F.one_hot(torch.tensor(annotation["category_id"], dtype=torch.long), num_classes=len(categories)).float().unsqueeze(0)
        #     if (i != 0 and annotation["image_id"] == annotations[i - 1]["image_id"]):
        #         self.img_cat[-1] = torch.cat([self.img_cat[-1], label], dim = 0)
        #     else:
        #         self.img_cat.append(label)
   
    """
    returns length of dataset
    """
    def __len__(self):
        return len(self.img_labels)
    """
    returns the tensors of an image and labels at the index
    """
    def __getitem__(self, idx):
        img_path = "training_data/data/" + self.img_dir[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256))
        image_tensor = TF.to_tensor(image)
        
        # a 1d tensor
        label_tensor = torch.tensor(self.img_labels[idx], dtype=torch.float)
        # add padding to label tensor
        label_tensor = F.pad(label_tensor, (0, 50 - label_tensor.size(0)), "constant", 0)
        # view as (10, 4)
        label_tensor = label_tensor.view(-1, 5)


        cat_tensor = torch.tensor(self.img_cat[idx], dtype=torch.float)  # shape: (num_objects, 60)
        # num_objects = cat_tensor.size(0)
        # cat_tensor = cat_tensor.view(-1)
        cat_tensor = F.pad(cat_tensor, (0, 600 - cat_tensor.size(0)), "constant", 0)
        cat_tensor = cat_tensor.view(-1, 60)
        # if num_objects < 10:
        #     padding = torch.zeros(10 - num_objects, 60)
        #     cat_tensor = torch.cat([cat_tensor, padding], dim=0)
        # elif num_objects > 10:
        #     cat_tensor = cat_tensor[:10]

        return image_tensor, label_tensor, cat_tensor


        
        


            