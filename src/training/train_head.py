import os
from typing import Optional
from copy import deepcopy

import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataloader.encodings import EncodedDataset
from src.models.CLIP import ZeroShotHead
from src.utils.config import Config
from src.dataloader.encodings import load_whole_dataset


NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSV_NAME = 'christian.csv'
LR = 1e-3
N_SPLITS = 5


classes =  NUM_CLASSES * ['This is a picture of a painting'] # classes is used to get a prior for the L2 regularization
base_model = ZeroShotHead(classes).to(DEVICE)
get_optimizer = lambda model: optim.Adam(model.parameters(), lr=LR)


X_all, y_all = load_whole_dataset(CSV_NAME, DEVICE, splits=['train', 'val'])


class CustomCriterion(nn.Module):

    def __init__(self, _lambda: float = 1e-1, base_model: Optional[ZeroShotHead]=None) -> None:
        super().__init__()

        self._lambda = _lambda
        self.cross_entropy = nn.CrossEntropyLoss()
        self.prior = base_model._weights if base_model else None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, model: Optional[ZeroShotHead]=None) -> torch.Tensor:
        loss = self.cross_entropy(y_pred, y_true)
        l2_reg = F.mse_loss(model._weights, self.prior if self.prior is not None else 0) if model is not None else 0
        return loss + self._lambda * l2_reg


def get_model() -> ZeroShotHead:
    model = deepcopy(base_model)
    model._weights.requires_grad = True
    model.temperature.requires_grad = True
    return model


def fit(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, optimizer: optim.Optimizer, criterion: nn.Module, min_grad: float=1e-3) -> None:
    
    pbar = tqdm.tqdm(desc='Training', leave=False)
    model.train()

    epoch = 0
    while True:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss: torch.Tensor = criterion(y_pred, y_train, model)
        loss.backward()
        optimizer.step()

        grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]), p=2)

        if epoch % 1000 == 0:
            pbar.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
        pbar.update()
        epoch += 1

        if grad_norm < min_grad:
            pbar.close()
            break


@torch.no_grad()
def evaluate(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    y_pred = model(X_val)
    accuracy = (y_pred.argmax(dim=1) == y_val).float().mean().item()
    loss: torch.Tensor = criterion(y_pred, y_val, model)
    return loss.item(), accuracy


K_folds = KFold(n_splits=N_SPLITS, shuffle=True)
losses = list()
accuracies = list()
criterion = CustomCriterion(base_model=base_model)
for split, (train_idx, val_idx) in tqdm.tqdm(enumerate(K_folds.split(X_all, y_all)), desc='Cross-validation', total=N_SPLITS):
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    model = get_model()
    optimizer = get_optimizer(model)

    fit(model, X_all, y_all, optimizer, criterion)
    loss, accuracy = evaluate(model, X_val, y_val, criterion)
    losses.append(loss)
    accuracies.append(accuracy)
    print(f'Split {split}: Loss: {loss}, Accuracy: {accuracy}')
