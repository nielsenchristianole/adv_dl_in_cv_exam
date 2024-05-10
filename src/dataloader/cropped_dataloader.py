import csv
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import numpy as np

import matplotlib.pyplot as plt


import torchvision.transforms as transforms

class CroppedDataloader(Dataset):
    def __init__(self, csv_file, image_dir, min_threshold, max_threshold, min_games, train=True, test_size=0.2):
        self.image_dir = image_dir
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_games = min_games
        self.data = self.load_data(csv_file)
        self.print_data_info()
        
        self.transform = transforms.Compose([transforms.Resize((224, 224))])
        
        if train:
            self.data, _ = train_test_split(self.data, test_size=test_size, random_state=42)
        else:
            _, self.data = train_test_split(self.data, test_size=test_size, random_state=42)
        
    def print_data_info(self):
        num_samples = len(self.data)
        num_classes = len(set([label for _, label in self.data]))
        class_counts = {label: 0 for _, label in self.data}
        for _, label in self.data:
            class_counts[label] += 1
        print(f"Number of samples: {num_samples}")
        print(f"Number of classes: {num_classes}")
        for label, count in class_counts.items():
            print(f"Number of samples in class {label}: {count}")

    def load_data(self, csv_file):
        data = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['discard'] == 'True':
                    continue
                if int(row['games']) < self.min_games:
                    continue
                elo = float(row['elo'])
                if elo < self.min_threshold:
                    label = 0
                elif elo > self.max_threshold:
                    label = 1
                else:
                    label = 2
                image_path = self.image_dir / (row['name'] + '.png')
                
                # check if the image exists
                if not os.path.exists(image_path):
                    continue
                
                data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        # image = Image.open(image_path).convert('RGB')
        data = np.load(Path("data/encoded/cropped") / (image_path.stem + ".npy"))
        
        data = torch.tensor(data).float()
        
        # image = np.array(image).transpose(2, 0, 1)
        # image = torch.tensor(image).float()
        # image = self.transform(image)
        
        # Preprocess the image if needed
        # ...

        return data, torch.tensor(label)
    
class GeneratedData(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.data = self.load_data(image_dir)
        
        self.transform = transforms.Compose([transforms.Resize((224, 224))])
        

    def load_data(self, image_dir):
        data = []
        
        for path in image_dir.glob("**/*.png"):
            data.append(path)
       
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        
        image = np.array(image).transpose(2, 0, 1)
        image = torch.tensor(image).float()
        image = self.transform(image)
        
        # Preprocess the image if needed
        # ...

        return image, 0

if __name__ == "__main__":
    
    # Define the hyperparameters
    csv_file = Path('data/elo_annotations/calle2.csv')
    image_dir = Path('data/cropped')
    min_threshold = 950.0
    max_threshold = 1050.0
    min_games = 6

    # Create the train and test dataloaders
    train_dataset = CroppedDataloader(csv_file, image_dir, min_threshold, max_threshold, min_games, train=True)
    test_dataset = CroppedDataloader(csv_file, image_dir, min_threshold, max_threshold, min_games, train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Iterate over the train dataloader
    for images, labels in train_dataloader:
        print("Train batch")
        # Print an image using matplotlib
        plt.imshow(images[0])
        plt.show()

        # Break out of the loop
        break
    
    # Iterate over the test dataloader
    for images, labels in test_dataloader:
        print("Test batch")
        # Print an image using matplotlib
        plt.imshow(images[0])
        plt.show()

        # Break out of the loop
        break
