import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, Compose

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import cv2

# training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
input_size = 3072  # 32*32*3
hidden_layer = 100
num_of_classes = 10  # CIFAR10 Dataset
epochs = 5
batch_size = 32
learning_rate = 0.05

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
    print(point[0].shape)
    break

print(train_dataset.class_to_idx)


# data visualization
data_points = iter(train_Loader)
samples, labels = data_points.__next__()
print(samples.shape, labels.shape)

plt.figure(figsize=(12,12), dpi=100)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(samples[i][0], cmap="gray")
    plt.title(labels[i])
# plt.show()

# Neural Network Model
class cifernet(nn.Module):
    def __init__(self, input, hidden, num_of_classes):
        super(cifernet, self).__init__()
        self.layer1 = nn.Linear(input,hidden)
        self.ReLU1 = nn.LeakyReLU()

        self.layer2 = nn.Linear(hidden, num_of_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.ReLU1(out)
        out = self.layer2(out)
        return out

# always send model to GPU
model = cifernet(input_size, hidden_layer, num_of_classes).to(device)
print(model)


# Loss and Optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for e in range(epochs):
    for i, (img, label) in enumerate(train_Loader):
        # 32, 3, 32, 32 -> 32, 3072
        img = img.reshape(-1, 3*32*32).to(device)
        label = label.to(device)

        # forward
        outputs = model(img)
        loss = criterion(outputs, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training
        if (i+1)%100 == 0:
            print(f"Num of epoch:{e}/{epochs} ---------- step: {i+1}---------- loss:{loss.item():.4f}")
print('Finished Training')


# Iterate over test data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for img, label in train_Loader:
        # 32, 3, 32, 32 -> 32, 3072
        img = img.reshape(-1, 3*32*32).to(device)
        label = label.to(device)

        outputs = model(img)
#
#       Actual predictions
        _, predictions = torch.max(outputs, 1)      # this returns value, index

        n_samples += label.shape[0]
        n_correct = (predictions == label).sum().item()

    acc = 100*n_correct/n_samples

    print(f"accuracy:{acc}")

















