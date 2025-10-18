import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torchvision.transforms import functional as TF
from PIL import Image

class bbox_dataset(Dataset):
    def __init__(self, annotations, images):
        # Initialize lists
        self.img_dir = [images[annotation["image_id"]]["file_name"] for annotation in annotations]
        self.img_labels = [annotation["bbox"] for annotation in annotations]
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256))
        image_tensor = TF.to_tensor(image)

        label_tensor = torch.tensor(self.img_labels[idx], dtype=torch.float)
        return image_tensor, label_tensor

class category_dataset(Dataset):
    def __init__(self, annotations, categories, images):
        # initialize lists
        self.img_dir = [images[annotation["image_id"]]["file_name"] for annotation in annotations]
        # one hot encoding
        # convert category_id into a 1x1 tensor
        self.img_labels = [F.one_hot(torch.tensor(annotation["category_id"], dtype=torch.long), num_classes=len(categories)).squeeze(0).float() for annotation in annotations]
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256))
        image_tensor = TF.to_tensor(image)
        
        return image_tensor, self.img_labels[idx]
        
        


            