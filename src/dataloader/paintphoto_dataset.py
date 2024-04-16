import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from src.utils.config import Config
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor

class PaintingPhotoDataset(Dataset):
    def __init__(self, root_dir, clip_model, processor, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class subfolders.
            clip_model (CLIPModel): Preloaded CLIP model to generate embeddings.
            processor (CLIPProcessor): Preloaded CLIP processor for image preprocessing.
            transform (callable, optional): Optional transform to be applied before processing for CLIP.
        """
        self.root_dir = root_dir
        self.clip_model = clip_model
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.processor = processor
        self.transform = transform
        self.images = []
        self.labels = []

        # List all directories in the root directory
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # Collect all image paths and their labels
        for cls_idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_folder, img_name))
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Prepare the image for CLIP
        inputs = self.processor(images=image, return_tensors="pt")

        # Generate the image embedding
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            embeddings = outputs.pooler_output

        # Get the label
        label = self.labels[index]

        return embeddings.squeeze(0), label

if __name__ == "__main__":
    # Initialize CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)

    # Define transformations
    transform = transforms.Compose([
    ])

    # Configurations and paths
    cfg = Config('configs/config.yaml')
    base_path = cfg.get('data', 'base_path')
    paintphoto_path_train = os.path.join(base_path, "paintphoto/train")
    paintphoto_val_path = os.path.join(base_path, "paintphoto/valid")

    # Initialize datasets
    train_dataset = PaintingPhotoDataset(paintphoto_path_train, clip_model, processor, transform)
    valid_dataset = PaintingPhotoDataset(paintphoto_val_path, clip_model, processor, transform)

    # Example: Create DataLoader for training dataset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # Example: Visualize a batch of images
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
