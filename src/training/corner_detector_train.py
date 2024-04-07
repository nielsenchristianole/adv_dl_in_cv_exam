import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from lightning import LightningModule, Trainer, loggers
from corner_detection_loader import CornersDataset

import torch.nn as nn
import torch.optim as optim

class CornerDetector(LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.fc = nn.Linear(512, 8)  # 8 outputs for corner coordinates

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = nn.MSELoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = nn.MSELoss()(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

if __name__ == '__main__':
    train_dataset = CornersDataset(is_train=True, transform=ToTensor())
    val_dataset = CornersDataset(is_train=False, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    corner_detector = CornerDetector()

    logger = loggers.TensorBoardLogger('logs', name='corner_detector')
    trainer = Trainer(logger=logger)
    trainer.fit(corner_detector, train_loader, val_loader)