import torch

# f = w*x
# f = 2*x           # w=2
X = torch.tensor([1, 2, 3, 5, 6], dtype=torch.float32)
Y = torch.tensor([1, 7, 10, 16, 20], dtype=torch.float32)
w = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)  # this allows for auto gradient calculation


# this is the prediction model

# This is the training function
def forward(x):
    return w * x


# loss function = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


print(f"Prediction before training at f(10) = {forward(10):.3f}")

# Training
lr = 0.01          # ideal learning rate
iters = 100

for epoch in range(iters):
    y_predict = forward(X)                  # prediction = forward pass
    l = loss(Y, y_predict)                  # loss
    l.backward()                            # gradients = backward pass = dl/dw

    # update weights
    with torch.no_grad():
        w -= lr * w.grad

    # zeroing the gradients
    w.grad.zero_()

    if epoch%10 == 0:
        print(f" epoch:{epoch+1}, w: {w:.3f},  loss: {l:.3f}")

print(f"after prediction at f(10): {forward(10):.3f}")