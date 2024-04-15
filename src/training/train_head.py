import os
from copy import deepcopy
from typing import Optional, Literal

import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.models.CLIP import ZeroShotHead
from src.utils.config import Config
from src.dataloader.encodings import EncodedDataset


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
    global base_model
    model = deepcopy(base_model)
    model._weights.requires_grad = True
    model.temperature.requires_grad = True
    return model


plt.ion()
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()


def update_plot(
    *,
    train_losses: list[float],
    validation_losses: list[float],
    train_accuracies: list[float],
    validation_accuracies: list[float],
    grad_norms: list[float],
    epochs: list[int]
) -> None:

    fig = plt.figure('training')
    fig.clear()

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, validation_losses, label='Test')
    ax.legend(loc='lower left')
    ax.set_title('Loss')
    ax.set_yscale('log')

    ax = fig.add_subplot(3, 1, 2)
    ax.set_title('Accuracy')
    ax.plot(epochs, train_accuracies, label='Train')
    ax.plot(epochs, validation_accuracies, label='Test')
    ax.legend(loc='lower left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    ax = fig.add_subplot(3, 1, 3)
    ax.set_title('Gradient Norm')
    ax.plot(epochs, grad_norms)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')

    fig.tight_layout()
    drawnow()


def train(
    *,
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int = 10_000
) -> None:
    model.train()

    train_losses = list()
    validation_losses = list()
    train_accuracies = list()
    validation_accuracies = list()
    grad_norms = list()
    epochs = list()

    global min_grad_global
    global update_plot_cycle_global

    for epoch in (pbar := tqdm.trange(num_epochs, desc='Training', leave=False)):

        optimizer.zero_grad()
        outputs = model(X_train)
        loss: torch.Tensor = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % update_plot_cycle_global == 0:
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]), p=2)

            train_loss, train_accuracy = eval(model=model, X=X_train, y=y_train, criterion=criterion)
            val_loss, val_accuracy = eval(model=model, X=X_test, y=y_test, criterion=criterion)
            model.train()

            train_losses.append(train_loss)
            validation_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(val_accuracy)
            grad_norms.append(grad_norm.item())
            epochs.append(epoch)

            update_plot(
                train_losses=train_losses,
                validation_losses=validation_losses,
                train_accuracies=train_accuracies,
                validation_accuracies=validation_accuracies,
                grad_norms=grad_norms,
                epochs=epochs
            )
            
            if grad_norm < min_grad_global:
                pbar.close()
                break


@torch.no_grad()
def eval(*, model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    
    out: torch.Tensor = model(X)
    loss: torch.Tensor = criterion(out, y)
    accuracy = (out.argmax(dim=1) == y).float().mean()
    
    return loss.item(), accuracy.item()


def main(
    lr: float = 1e-3,
    cv_splits: int = 5,
    min_grad: float = 1e-5,
    update_plot_cycle: int = 100,
    num_epochs: int = 100_000,
    lambdas: list[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    csv_name: str = 'calle2.csv',
    device: Optional[Literal['cuda', 'cpu']] = None,
    output_path: str = 'models/head.pth'
):
    global min_grad_global
    global update_plot_cycle_global
    global base_model
    
    min_grad_global = min_grad
    update_plot_cycle_global = update_plot_cycle

    # load data
    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'encoded_path')
    ann_folder = cfg.get('data', 'annotations_path')

    csv_path = os.path.join(ann_folder, csv_name)

    dataset = EncodedDataset(csv_file=csv_path, root_dir=data_folder)
    dataloader = DataLoader(dataset.get_dataset_split_subset(['train', 'val']), shuffle=True)
    data = list(dataloader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)

    X = torch.concat([d[0] for d in data]).to(torch.float).to(device)
    y = torch.concat([d[1] for d in data]).to(torch.long).to(device)

    num_classes = len(torch.unique(y))

    base_model = ZeroShotHead(num_classes * ['This is a picture of a painting']).to(device)

    cross_validation = KFold(n_splits=cv_splits, shuffle=True)

    for train_index, test_index in tqdm.tqdm(cross_validation.split(X), desc='Outer CV', total=cv_splits):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        
        gen_losses = len(lambdas) * [0]
        gen_accuracies = len(lambdas) * [0]

        for idx, _lambda in enumerate(lambdas):
            criterion = CustomCriterion(_lambda=_lambda, base_model=base_model)
            model = get_model()
            optimizer = Adam(model.parameters(), lr=lr)

            train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=num_epochs
            )

            loss, accuracy = eval(model=model, X=X_test, y=y_test, criterion=criterion)
            gen_losses[idx] += loss * len(test_index) / len(train_index)
            gen_accuracies[idx] += accuracy * len(test_index) / len(train_index)


    best_lambda = lambdas[gen_accuracies.index(max(gen_accuracies))]
    model = get_model()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CustomCriterion(_lambda=best_lambda, base_model=base_model)

    dataloader = DataLoader(dataset.get_dataset_split_subset('test'))
    test_data = list(dataloader)
    X_test = torch.concat([d[0] for d in test_data]).to(torch.float).to(device)
    y_test = torch.concat([d[1] for d in test_data]).to(torch.long).to(device)

    train(
        model=model,
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs
    )

    test_loss, test_accuracy = eval(model=model, X=X_test, y=y_test, criterion=criterion)
    train_loss, train_accuracy = eval(model=model, X=X, y=y, criterion=criterion)

    print(f'Best lambda: {best_lambda}')
    print(f'Final test loss: {test_loss}')
    print(f'Final test accuracy: {test_accuracy}')
    print(f'Final train loss: {train_loss}')
    print(f'Final train accuracy: {train_accuracy}')

    torch.save(model.state_dict(), output_path)



if __name__ == '__main__':
    import tyro
    tyro.cli(main)
