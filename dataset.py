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
    def __init__(self, annotations, images):
        # Initialize lists
        self.img_dir = [images[annotation["image_id"]]["file_name"] for annotation in annotations]
        self.img_labels = []
        # Append all bbox's to label lists, but make sure each one is scaled to 256x256 pixels
        for annotation in annotations:
            scale_x = 256 / images[annotation["image_id"]]["width"]
            scale_y = 256 / images[annotation["image_id"]]["height"]
            x, y, w, h = annotation["bbox"]
            bbox_scaled = [
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y
            ]
            self.img_labels.append(bbox_scaled)        
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

        label_tensor = torch.tensor(self.img_labels[idx], dtype=torch.float)
        
        return image_tensor, label_tensor

class category_dataset(Dataset):
    """
    creates a Dataset of images and uses categories as labels

    Attributes:
        img_dir (list): List of image paths
        img_labels (list): List of category labels as one hot encodings
    """
    def __init__(self, annotations, categories, images):
        # initialize lists
        self.img_dir = [images[annotation["image_id"]]["file_name"] for annotation in annotations]
        # one hot encoding
        # convert category_id into a 1x1 tensor
        self.img_labels = [F.one_hot(torch.tensor(annotation["category_id"], dtype=torch.long), num_classes=len(categories)).squeeze(0).float() for annotation in annotations]
    """
    return length of dataset
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
        
        return image_tensor, self.img_labels[idx]
        
        


            