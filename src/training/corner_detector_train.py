import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, mobilenet_v3_large

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataloader.corner_detection_loader import CornerDataset

import cv2
import numpy as np
import os

import torch.nn as nn
import torch.optim as optim

class CornerDetector(L.LightningModule):
    
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
            
    def __init__(self, scale_to = 256):
        super().__init__()
        self.save_hyperparameters()
        
        self.scale_to = scale_to
        
        self.mobile_net = mobilenet_v3_large(pretrained=True)
        last_channel = self.mobile_net.classifier[-1].in_features
        self.mobile_net.classifier[-1] = nn.Linear(last_channel, 8)

        # Create train loss file if it doesn't exist
        if not os.path.exists(self.train_loss_file):
            with open(self.train_loss_file, 'w') as f:
                pass

        # Create val loss file if it doesn't exist
        if not os.path.exists(self.val_loss_file):
            with open(self.val_loss_file, 'w') as f:
                pass

    def forward(self, x):
        return self.mobile_net(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = nn.MSELoss()(outputs[:-1], targets[:-1])
        
        if batch_idx == 0:
            with open('train_loss.txt', 'a') as f:
                f.write(loss.item())
            self.log('train/loss', loss)
            self._log_predictions(batch, outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = nn.MSELoss()(outputs[:-1], targets[:-1])
        
        if batch_idx <= 10:
            with open('val_loss.txt', 'a') as f:
                f.write(loss.item())
            self.log('val/loss', loss)
            self._log_predictions(batch, outputs, "val")
            
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def _log_predictions(self, batch, outputs, prefix : str):
        
        def draw_corners(img, corners, color):
            corners *= self.scale_to
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
            
            img = img * self.IMAGENET_STD + self.IMAGENET_MEAN
            img = np.clip(img, 0, 1)
            img = np.ascontiguousarray(img)
            
            draw_corners(img, pred, (0, 1, 0))
            draw_corners(img, target, (1, 0, 0))
            
            img = (img * 255).astype(np.uint8)
            images[i] = img.transpose(2, 0, 1)
        self.logger.experiment.add_images(f"{prefix}",images.astype(np.uint8), self.current_epoch) # type: ignore

if __name__ == '__main__':
    
    name = input("Insert name of training:")
    
    scale_to = 256
    batch_size = 2
    
    train_dataset = CornerDataset(is_train=True, scale_to=scale_to, train_split = 0.85)
    val_dataset = CornerDataset(is_train=False, scale_to=scale_to, train_split=0.85)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=5, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=3, pin_memory=True, persistent_workers=True)

    corner_detector = CornerDetector(scale_to)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',  # The validation metric to monitor
        dirpath='model_checkpoints',  # Directory where the checkpoints will be saved
        filename=f"{name}",  # The prefix for the checkpoint file name
        save_top_k=5,  # Save only the best model
        mode='min',  # 'min' means that lower 'val_loss' is better
    )
    logger = TensorBoardLogger("lightning_logs", name=f"{name}")
    trainer = L.Trainer(logger=logger,
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=1)
    trainer.fit(corner_detector, train_loader, val_loader)