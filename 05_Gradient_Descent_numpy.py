# Manual way of counting gradient descent

import numpy as np

# f = w*x
# f = 2*x           # w=2
X = np.array([1, 2, 3, 5, 6], dtype=np.float32)
Y = np.array([1, 4, 9, 25, 36], dtype=np.float32)
w = 0.5


# this is the prediction model
# This is the training function
def forward(x):
    return w * x * x


# loss function = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# gradient calculation
# MSE = 1/N (w*x - y)**2
# dJ/dw = 1/N 2x (w*x-y)

def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


print(f"Prediction before training at f(10) = {forward(10):.3f}")

# Training
lr = 0.001          # ideal learning rate
iters = 5

for epoch in range(iters):
    y_predict = forward(X)                  # prediction
    l = loss(Y,y_predict)                   # loss
    grd = gradient(X,Y,y_predict)           # gradients
    w -= lr*grd                             # update weights

    if epoch%1 == 0:
        print(f" epoch:{epoch+1}, w: {w:.3f},  loss: {l:.3f}")

print(f"after prediction at f(10): {forward(10):.3f}")