import numpy as np
import requests
import torch


import os
import pickle
import tarfile

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True):
        if not os.path.exists("cifar-10-python.tar.gz"):
            print("Downloading dataset...")
            r = requests.get(CIFAR10_URL)
            with open("cifar-10-python.tar.gz", "wb") as f:
                f.write(r.content)
        if not os.path.exists("cifar-10-batches-py"):
            print("Extracting dataset...")
            tar = tarfile.open("cifar-10-python.tar.gz")
            tar.extractall()
            tar.close()
        self.data = []
        self.labels = []
        if train:
            for i in range(1, 6):
                with open(f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                batch_data = data[b"data"]
                batch_data = batch_data.reshape(-1, 3, 32, 32)
                self.data.append(batch_data)
                self.labels += data[b"labels"]
        else:
            with open(f"cifar-10-batches-py/test_batch", "rb") as f:
                data = pickle.load(f, encoding="bytes")
            batch_data = data[b"data"]
            batch_data = batch_data.reshape(-1, 3, 32, 32)
            self.data.append(batch_data)
            self.labels += data[b"labels"]
        self.data = np.concatenate(self.data, axis=0)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    @property
    def classes(self):
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
