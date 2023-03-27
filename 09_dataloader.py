# This is followed from -> https://youtu.be/_BxXrFStVOQ

import numpy as np
import pandas as pd
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.datasets import load_breast_cancer

breast_data = load_breast_cancer()     # this is in numpy array shape

# converting into pandas dataframe
df = pd.DataFrame(data = breast_data.data, columns=breast_data.feature_names)
df["target"] = pd.Series(breast_data.target)


train = df.iloc[:,0:30]
target = df.iloc[:,[30]]
# print(train)
# df.to_csv('breast_cancer.csv', index=False) # saving as csv


# converting pandas to tensor
class breastTensor(Dataset):

    def __init__(self, train, target):
        self.train = train
        self.target = target

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        dataPoint = torch.tensor(self.train.iloc[idx])
        outcome = torch.tensor(self.target.iloc[idx])
        return dataPoint, outcome

breast_data = breastTensor(train,target)        # this is the tensor dataset

# dataloader

dl = DataLoader(
    dataset=breast_data,
    batch_size=16,
    shuffle=True,
)

for point in dl:
    # point[0] = train, point[1] = target
    print(point[0].shape)
    break
