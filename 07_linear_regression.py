import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# generating a dataset
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=100)
# converting to torch
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))  # this is a row vector, we want a column vector

y = y.view(y.shape[0], 1)
n_samples, n_features = x.shape

# model
model = nn.Linear(
    in_features=n_features,
    out_features=1
)

# loss
criterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training
epochs = 1000
for e in range(epochs):
    # forward pass
    y_pred = model(x)
    # backward pass
    loss = criterion(y_pred, y)
    loss.backward()
    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if e % 100 == 0:
        print(f"epoch = {e + 1} loss = {loss:.3f}")


# plotting
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, "ro")
plt.plot(x_numpy, predicted, "b")