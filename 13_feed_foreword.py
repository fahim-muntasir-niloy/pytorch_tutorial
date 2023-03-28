import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, Compose

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# training on GPU
device = "gpu" if torch.cuda.is_available() else "cpu"

# Hyper parameters
input_size = 1024  # 32*32
hidden_layer = 100
num_classes = 10  # CIFAR10 Dataset
epochs = 20
batch_size = 32
learning_rate = 0.001

# CIFAR10 Dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="G:\\Work\\python\\pytorch_patrick_loeber",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="G:\\Work\\python\\pytorch_patrick_loeber",
    train=False,
    transform=transforms.ToTensor()
)
# train_leader
train_Loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_Loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

for point in train_Loader:
    # point[0] = train, point[1] = target
    for samples, labels in train_dataset:
        samples = point[0]
        labels = point[1]

    print(samples.shape)
    break

print(train_dataset.class_to_idx)

"""
Data vizualization
"""