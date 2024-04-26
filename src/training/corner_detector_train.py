import torch
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

import lightning as L
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from src.dataloader.corner_detector_loader import CornerDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_optimizer import create_optimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    learning_rate_max: float = 1e-2
    learning_rate_min: float = 1e-3
    learning_rate_half_period: int = 2000
    learning_rate_mult_period: int = 2
    learning_rate_warmup_max: float = 4e-2
    learning_rate_warmup_steps: int = 1000
    weight_decay: float = 1e-6

class CornerDetector(LightningModule):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=False)
        self.fc = nn.Linear(2048, 8)  # 8 outputs for corner coordinates
        
        self.config = TrainConfig()

    def forward(self, x):
        x = self.inception(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = nn.MSELoss()(outputs[:-1], targets[:-1])
        
        if batch_idx == 0:
            self.log('train/loss', loss)
            self._log_predictions(batch, outputs, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = nn.MSELoss()(outputs[:-1], targets[:-1])
        
        if batch_idx <= 10:
            self.log('val/loss', loss)
            self._log_predictions(batch, outputs, "val")
            
        return loss
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = create_optimizer(
            self, # type: ignore
            "adan",
            lr=self.config.learning_rate_max,
            weight_decay=self.config.weight_decay,
            use_lookahead=True,
            use_gc=True,
            eps=1e-6
        )

        # NOTE: Must instantiate cosine scheduler first,
        #  because super scheduler mutates the initial learning rate.
        lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.config.learning_rate_half_period,
            T_mult=self.config.learning_rate_mult_period,
            eta_min=self.config.learning_rate_min
        )
        lr_super = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config.learning_rate_warmup_max,
            total_steps=self.config.learning_rate_warmup_period,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[lr_super, lr_cosine], # type: ignore
            milestones=[self.config.learning_rate_warmup_period],
        )

        return { # type: ignore
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }

if __name__ == '__main__':
    name = input("Insert name of training:")
    
    scale_to = 256
    batch_size = 16
    
    train_dataset = CornerDataset(is_train=True, scale_to=scale_to)
    val_dataset = CornerDataset(is_train=False, scale_to=scale_to)
    
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