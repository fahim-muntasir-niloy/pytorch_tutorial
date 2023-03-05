import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [7.5], [9]], dtype=torch.float32)

x_test = torch.tensor([20], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features  # here they are same, but this case is not always

# training func
model = nn.Linear(in_features=input_size,
                  out_features=output_size)

print(f"predictions before training = {model(x_test).item():.3f}")

iters = 1000
loss = nn.MSELoss()
learning_rate = 0.001

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# training
for e in range(iters):
    y_pred = model(x)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if e % 333 == 0:
        print(f"epoch = {e + 1} loss = {l:.3f} Predicted value = {model(x_test).item():.3f}")

print(f"final predicted value = {model(x_test).item():.3f}")
