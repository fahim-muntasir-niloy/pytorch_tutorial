import numpy as np
import pandas as pd
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import sklearn
from sklearn.datasets import load_breast_cancer

breast_data = load_breast_cancer()  # this is in numpy array shape

# converting into pandas dataframe
df = pd.DataFrame(data=breast_data.data, columns=breast_data.feature_names)
df["target"] = pd.Series(breast_data.target)

train = df.iloc[:, 0:30]
target = df.iloc[:, [30]]


# converting pandas to tensor
class breastTensor(Dataset):

    def __init__(self, train, target, transform=None):
        self.train = train
        self.target = target
        self.transform = transform

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        sample = self.train.iloc[idx], self.target.iloc[idx]
        # dataPoint = torch.tensor(self.train.iloc[idx])
        # outcome = torch.tensor(self.target.iloc[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample


# transformation
class toTensor:
    def __call__(self, sample):
        datapoints, outcome = sample
        return torch.tensor(datapoints), torch.tensor(outcome)


class mulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        datapoints, outcomes = sample
        datapoints *= self.factor
        return datapoints, outcomes


"""
for multiple transforms, we use compose method from torch to make them into one.
This is a list of all custom transformers
"""
composeTransformer = Compose([toTensor(),
                              mulTransform(5)])

breast_data = breastTensor(train, target, transform=composeTransformer)  # this is the tensor dataset

# dataloader
dl = DataLoader(
    dataset=breast_data,
    batch_size=16,
    shuffle=True,
)

for point in dl:
    # point[0] = train, point[1] = target
    print(point[0][2], point[1][2])
    break
