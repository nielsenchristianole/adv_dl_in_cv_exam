import os
from copy import deepcopy
from typing import Optional, Literal
from pathlib import Path
from tabulate import tabulate

import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from src.models.CLIP import ClipHead, ClipHeadTypes
from src.utils.config import Config
from src.dataloader.encodings import EncodedDataset


CFG = Config('configs/config.yaml')

class CustomCriterion(nn.Module):
    """
    Cross entropy loss with l2-regularization.

    If use_non_zero_prior is True, the prior model is used to compute the l2-regularization.
    """

    def __init__(self, _lambda: float = 1e-1, base_model: Optional[ClipHead]=None, use_non_zero_prior: bool=False) -> None:
        super().__init__()

        self._lambda = _lambda
        self.cross_entropy = nn.CrossEntropyLoss()
        self.prior = base_model
        self.use_non_zero_prior = use_non_zero_prior

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, model: ClipHead) -> torch.Tensor:
        loss = self.cross_entropy(y_pred, y_true)
        if self.use_non_zero_prior and self.prior is not None:
            l2_reg = 0
            for p1, p2 in zip(model.parameters(), self.prior.parameters()):
                l2_reg += (p1 - p2).square().sum()
        else:
            l2_reg = 0
            for p in model.parameters():
                l2_reg += p.square().sum()
        return loss + self._lambda * l2_reg.sqrt()


def get_model() -> ClipHead:
    """Return a new model instance."""
    global base_model
    model = deepcopy(base_model).train()
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
    epochs: list[int],
    save: bool = False
) -> None:
    """Update the training plot. If save is True, save the plot.
    if show_plot is False, do not compute the plot."""
    
    global show_plot_global
    if not show_plot_global or not save:
        return

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
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)

    ax = fig.add_subplot(3, 1, 3)
    ax.set_title('Gradient Norm')
    ax.plot(epochs, grad_norms)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')

    if save is not None:
        global output_path_global
        save_path = output_path_global.with_suffix('.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot_global:
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
    num_epochs: int,
    save_best: bool = False,
    early_stopping: Optional[int] = None
) -> tuple[float, float]:
    model.train()

    # init tracked vals
    train_losses = list()
    validation_losses = list()
    train_accuracies = list()
    validation_accuracies = list()
    grad_norms = list()
    epochs = list()

    global min_grad_global
    global update_plot_cycle_global

    # track best values
    best_accuracy = -float('inf')
    best_loss = float('inf')
    last_best_epoch = 0

    for epoch in (pbar := tqdm.trange(num_epochs, desc='Training', leave=save_best)):

        # forward pass
        optimizer.zero_grad()
        outputs = model(X_train)
        loss: torch.Tensor = criterion(outputs, y_train, model)
        loss.backward()
        optimizer.step()

        if epoch % update_plot_cycle_global == 0:
            # compute metrics

            # used for early stopping
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]), p=2).item()

            # evaluate model
            train_loss, train_accuracy = eval(model=model, X=X_train, y=y_train, criterion=criterion)
            val_loss, val_accuracy = eval(model=model, X=X_test, y=y_test, criterion=criterion)

            # update progress bar with metrics
            pbar.set_postfix(
                Tloss=train_loss,
                Vloss=val_loss,
                Tacc=train_accuracy,
                Vacc=val_accuracy,
                grad=grad_norm
            )
            
            # save metrics for plotting
            train_losses.append(train_loss)
            validation_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(val_accuracy)
            grad_norms.append(grad_norm)
            epochs.append(epoch)

            # check if best model
            if val_accuracy > best_accuracy:
            # if val_loss < best_loss:
                best_accuracy = val_accuracy
                best_loss = val_loss
                last_best_epoch = epoch
                if save_best:
                    # save model
                    global output_path_global
                    torch.save(model.state_dict(), output_path_global.with_stem(output_path_global.stem + '_best'))
            if (early_stopping is not None) and \
                (epoch - last_best_epoch > early_stopping):
                # early stopping
                pbar.close()
                break
            
            model.train()

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
    
    update_plot(
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_accuracies=train_accuracies,
        validation_accuracies=validation_accuracies,
        grad_norms=grad_norms,
        epochs=epochs,
        save=save_best # update plot and save
    )

    return min(validation_losses), max(validation_accuracies)


@torch.no_grad()
def eval(*, model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[float, float]:
    """Evaluate the model on the given data."""

    model.eval()
    
    out: torch.Tensor = model(X)
    loss: torch.Tensor = criterion(out, y, model)
    accuracy = (out.argmax(dim=1) == y).float().mean()
    
    return loss.item(), accuracy.item()


def main(
    *,
    lr: float = 5e-3, # learning rate
    cv_splits: int = 5, # number of cross-validation splits
    min_grad: float = 0, # minimum gradient norm to stop training
    update_plot_cycle: int = 100, # update plot and metrics every n epochs
    num_epochs: int = 100_000, # maximum number of epochs
    lambdas: list[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], # l2-regularization values (differnt models)
    csv_name: str = 'wikiart_train.csv', # training annotations file
    test_csv_name: Optional[str] = 'wikiart_test.csv', # test annotations file, if None, use the test split from the training annotations
    device: Optional[Literal['cuda', 'cpu']] = None, # device to use
    output_path: str = 'models/head.pth', # output path for the model
    head_type: ClipHeadTypes = ClipHeadTypes['linear'], # which head to use
    root_dir: str = 'data/wikiart_encodings', # where to find the encodings
    early_stopping: Optional[int] = 1500, # early stopping if no improvement after n epochs
    show_plot: bool = False, # show the training plot while training
    mode: Literal['train', 'eval'] = 'eval' # train or evaluate the model
) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    print(f'Using device: {device}')

    global min_grad_global
    global update_plot_cycle_global
    global base_model
    global output_path_global
    global show_plot_global
    
    min_grad_global = min_grad
    update_plot_cycle_global = update_plot_cycle
    output_path_global = Path(output_path)
    show_plot_global = show_plot

    # load data into single tensor
    cfg = Config('configs/config.yaml')
    ann_folder = cfg.get('data', 'annotations_path')

    csv_path = os.path.join(ann_folder, csv_name)

    dataset = EncodedDataset(csv_file=csv_path, root_dir=root_dir)
    train_splits = ['train', 'val'] if test_csv_name is None else ['train', 'val', 'test']
    dataloader = DataLoader(dataset.get_dataset_split_subset(train_splits), shuffle=True)
    data = list(dataloader)

    X = torch.concat([d[0] for d in data]).to(torch.float).to(device)
    y = torch.concat([d[1] for d in data]).to(torch.long).to(device)

    num_classes = len(torch.unique(y))

    # use the selected model head
    base_model = head_type.value(num_classes * ['This is a picture of a painting']).to(device)

    cross_validation = KFold(n_splits=cv_splits, shuffle=True)

    gen_losses = len(lambdas) * [0]
    gen_accuracies = len(lambdas) * [0]

    total_datapoints = len(y)

    # k-fold cross-validation
    for train_index, test_index in tqdm.tqdm(cross_validation.split(X), desc='Folds', total=cv_splits):
        if mode == 'eval':
            break
        # split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train each model on data split
        for idx, _lambda in enumerate(tqdm.tqdm(lambdas, desc='Models', leave=False)):
            criterion = CustomCriterion(_lambda=_lambda, base_model=base_model)
            model = get_model()
            optimizer = Adam(model.parameters(), lr=lr)

            # training
            loss, accuracy = train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=num_epochs,
                early_stopping=early_stopping
            )

            # update generalization metrics
            metric_weights = len(test_index) / total_datapoints
            gen_losses[idx] += loss * metric_weights
            gen_accuracies[idx] += accuracy * metric_weights

    # print results
    best_lambda = lambdas[gen_accuracies.index(max(gen_accuracies))]
    print(f'Best lambda: {best_lambda}')
    print(tabulate(zip(lambdas, gen_losses, gen_accuracies), headers=['Lambda', 'Loss', 'Accuracy']))

    # train the best model on the whole dataset
    model = get_model()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CustomCriterion(_lambda=best_lambda, base_model=base_model)

    # load test data, either from the test split or from training csv
    if test_csv_name is not None:
        dataset = EncodedDataset(csv_file=os.path.join(ann_folder, test_csv_name), root_dir=root_dir)
        dataloader = DataLoader(dataset)
    else:
        dataloader = DataLoader(dataset.get_dataset_split_subset('test'))
    test_data = list(dataloader)
    X_test = torch.concat([d[0] for d in test_data]).to(torch.float).to(device)
    y_test = torch.concat([d[1] for d in test_data]).to(torch.long).to(device)

    if mode == 'train':
        # train the best model
        train(
            model=model,
            X_train=X,
            y_train=y,
            X_test=X_test,
            y_test=y_test,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            save_best=True,
            early_stopping=early_stopping
        )
    elif mode == 'eval':
        # load the best model
        model.load_state_dict(torch.load(output_path_global.with_stem(output_path_global.stem + '_best')))

    # evaluate the model
    test_loss, test_accuracy = eval(model=model, X=X_test, y=y_test, criterion=criterion)
    train_loss, train_accuracy = eval(model=model, X=X, y=y, criterion=criterion)

    # save results
    msg = (f'Best lambda: {best_lambda}\n'
           f'Final test loss: {test_loss}\n'
           f'Final test accuracy: {test_accuracy}\n'
           f'Final train loss: {train_loss}\n'
           f'Final train accuracy: {train_accuracy}')
    print(msg)

    if mode == 'train':
        with open(output_path_global.with_suffix('.txt'), 'w') as f:
            f.write(msg)

        torch.save(model.state_dict(), output_path_global.with_stem(output_path_global.stem + '_last'))

    # confusion matrix
    model.eval()
    y_pred = model(X_test).argmax(dim=1)
    cm = confusion_matrix(y_test.cpu(), y_pred.cpu(), normalize='true')
    plt.figure()
    plt.imshow(cm, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path_global.with_stem(output_path_global.stem + '_cm').with_suffix('.pdf'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    """Runs the main function with the tyro CLI. It works like argparse. Run with --help to see the options."""
    import tyro
    tyro.cli(main)
