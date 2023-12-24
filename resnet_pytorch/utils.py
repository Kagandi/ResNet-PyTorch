# functions to show an image


import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from itertools import product
import matplotlib.pyplot as plt


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


def lr_finder(model, criterion, optimizer, train_loader, device) -> float:
    losses = []
    lrs = torch.logspace(-6, 1, 100)
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        break
    for lr in tqdm(lrs):
        optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs.numpy(), losses)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()
    return lrs[np.argmin(losses)]



def denormalize(
    images: torch.Tensor, means: list[float], stds: list[float]
) -> torch.Tensor:
    """Denormalize images.
    :param images: images to be denormalized
    :param means: means used for normalization
    :param stds: standard deviations used for normalization
    :return: denormalized images
    """
    means = torch.tensor(means).reshape(3, 1, 1)
    stds = torch.tensor(stds).reshape(3, 1, 1)
    return images * stds + means


def show_predictions(
    predictions: np.array, labels: np.array, images: torch.Tensor, classes: list[str]
) -> None:
    """Show predictions vs true labels.
    :param predictions: predictions
    :param labels: true labels
    :param images: images
    :param classes: list of classes
    :return: None
    """
    idx2lable = classes

    fig, axs = plt.subplots(ncols=5, nrows=5, squeeze=False, figsize=(10, 10))
    grid_idx = list(product(range(5), repeat=2))
    for i, img in enumerate(images):
        x, y = grid_idx[i]
        img = img.detach()
        img = F.to_pil_image(img)
        axs[x, y].imshow(np.asarray(img))
        axs[x, y].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[x, y].set_title(
            f"{idx2lable[predictions[i]]} - {idx2lable[labels[i]]}", color="red"
        )

    fig.suptitle("Predictions vs True Label")

    plt.show()
