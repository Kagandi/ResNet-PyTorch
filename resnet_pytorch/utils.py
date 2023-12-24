# functions to show an image


import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class EarlyStopping(object):
    def __init__(
        self,
        patience: int = 10,
        path: Path = Path("."),
        verbose: bool = False,
        delta: float = 0,
    ):
        """Early stops the training if validation loss doesn't improve after a given patience.
        :param patience: how long to wait after last time validation loss improved.
        :param path: where to save the model
        :param verbose: if True, prints a message for each validation loss improvement.
        :param delta: minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.Inf
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model, f"{self.path}/checkpoint.pt")
        self.val_loss_min = val_loss
