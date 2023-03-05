import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# dataset
breast_cancer_data = datasets.load_breast_cancer()
X,y = breast_cancer_data.data, breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)