from resnet_pytorch.data import CIFAR10Dataset
from resnet_pytorch import ResNet, ResidualBlock, ResNet9
from resnet_pytorch.utils import imshow, EarlyStopping
from resnet_pytorch.models import resnet18
from resnet_pytorch.augmentaions import Flip, Rotate, Mirror, Contrast, Brightness
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import time
import numpy as np

if __name__ == "__main__":
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            Flip(),
            Rotate(),
            Mirror(),
            Contrast(),
            Brightness(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
        ]
    )


    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ]
    )
    cifar10_dataset = CIFAR10Dataset(preprocess)
    cifar10_dataloader = torch.utils.data.DataLoader(
        cifar10_dataset, batch_size=128, shuffle=True
    )
    # Stratified split into train test, and validation sets
    train_indices, val_indices = train_test_split(
        list(range(len(cifar10_dataset))),
        test_size=0.2,
        stratify=cifar10_dataset.labels,
    )
    BS = 1024
    train_dataset = Subset(cifar10_dataset, train_indices)
    val_dataset = Subset(cifar10_dataset, val_indices)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BS, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BS, shuffle=True
    )
    # imshow(train_dataset[0][0])
    # imshow(train_dataset[0][0])

    # imshow(train_dataset[0][0])

    # resnet = resnet18(in_channels=16, num_classes=10)
    resnet =  ResNet9(3, 10)
    EPOCHS = 10
    LR = 0.4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
    )
    early_stoping = EarlyStopping(patience=7, verbose=False, path=f"models", delta=0.005)

    for epoch in tqdm(range(EPOCHS)):
        total_loss = 0.0
        avg_loss = 0.0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for i, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(
                    train_loss=avg_loss,
                    val_loss=np.nan,
                    val_accuracy=np.nan,
                )

                optimizer.zero_grad()

                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)

                if val_dataloader is not None and i == len(train_dataloader) - 1:
                    with torch.no_grad():
                        val_loss = 0.0
                        val_score = 0.0
                        for i, (inputs, labels) in enumerate(val_dataloader):
                            outputs = resnet(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            val_score += (
                                (outputs.argmax(dim=1) == labels).float().mean().item()
                            )
                        val_loss /= len(val_dataloader)
                        val_score /= len(val_dataloader)
                        tepoch.set_postfix(
                            train_loss=avg_loss,
                            val_loss=val_loss,
                            val_accuracy=val_score,
                        )
                        time.sleep(0.5)
            scheduler.step()
            early_stoping(val_score, resnet)
            if early_stoping.early_stop:
                print("Early stopping")
                break
    resnet = torch.load(f"models/checkpoint.pt")


    resnet.eval()
    # PATH = "./cifar_net.pth"
    # torch.save(resnet.state_dict(), PATH)
    cifar10_dataset_test = CIFAR10Dataset(transform_test, train=False)
    cifar10_dataloader_test = torch.utils.data.DataLoader(
        cifar10_dataset_test, batch_size=32, shuffle=True
    )
    predict = []
    trues = []
    with torch.no_grad():
        test_loss = 0.0
        test_score = 0.0
        for i, (inputs, labels) in enumerate(
            tqdm(cifar10_dataloader_test, unit="batch", desc="Test")
        ):
            trues.append(labels)
            outputs = resnet(inputs)
            predict.append(outputs.argmax(dim=1))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_score += (outputs.argmax(dim=1) == labels).float().mean().item()
        test_loss /= len(cifar10_dataloader_test)
        test_score /= len(cifar10_dataloader_test)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_score}")

        # if i % 100 == 99:
        #     print(f'[{epoch + 1}, {i + 1}] loss: {total_loss / 100}')
        #     total_loss = 0.0
