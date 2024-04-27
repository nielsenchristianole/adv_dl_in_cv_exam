
from src.dataloader.corner_detection_loader import CornerDataset
from src.training.corner_detector_train import CornerDetector
import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def show_predictions(batch, outputs, prefix : str, scale_to=256):
        
    def draw_corners(img, corners, color):
        corners *= scale_to
        corners = corners.reshape(4, 2)
        
        for i in range(len(corners)):
            cv2.line(img, (int(corners[i, 0]), int(corners[i, 1])), (int(corners[(i+1) % 4, 0]), int(corners[(i+1) % 4, 1])), color, 2)
            cv2.putText(img, f"{i}", (int(corners[i, 0]), int(corners[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    
    imgs = batch[0].cpu().numpy()
    corners_target = batch[1].cpu().numpy()
    outputs = outputs.cpu().detach().numpy()
    
    images = np.zeros_like(imgs)
    for i, (img, pred, target) in enumerate(zip(imgs, outputs, corners_target)):
        img = img.transpose(1, 2, 0)
        
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)
        img = np.ascontiguousarray(img)
        
        draw_corners(img, pred, (0, 1, 0))
        draw_corners(img, target, (1, 0, 0))
        
        img = (img * 255).astype(np.uint8)
        
        plt.imshow(img)
        plt.show()
    
        

# Load the pretrained model
model = CornerDetector.load_from_checkpoint('model_checkpoints/inception.ckpt', map_location='cpu')

# Load the test dataset
test_dataset = CornerDataset(scale_to=512, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# Set the model to evaluation mode
model.eval()

# # Iterate over the test dataset and make predictions
# for batch in test_dataloader:
#     inputs, targets = batch
orig = cv2.imread('data/morten.png')
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
img : np.ndarray = cv2.resize(orig, (512, 512))
img = img / 255
img = (img - IMAGENET_MEAN) / IMAGENET_STD
img = img.transpose(2, 0, 1)

import matplotlib.pyplot as plt

input = torch.tensor(img, dtype=torch.float32)[None,...]

# Forward pass
with torch.no_grad():
    outputs = model(input)
    
# scale the bbox from 0-1 to orig.shape
outputs = outputs[0].cpu().numpy()
w, h = orig.shape[1], orig.shape[0]
outputs = outputs * np.array([w, h, w, h, w, h, w, h])

# plot the orignal image and the predicted bbox

plt.imshow(orig)
plt.plot([outputs[0], outputs[2], outputs[4], outputs[6], outputs[0]], [outputs[1], outputs[3], outputs[5], outputs[7], outputs[1]], 'o-')
plt.show()
    # show_predictions(batch, outputs, "test", scale_to=512)