import torch

x = torch.tensor(3.0)
y = torch.tensor(4.0)

w = torch.tensor(4.0, requires_grad=True)

# forward pass and loss compute
y_h = w*x
loss = (y_h-y)**2
print(f"loss = {loss}")

# backward pass
loss.backward()
print(f"weight gradient = {w.grad}")

#%%
"""
Next steps:
1. update weights
2. more forward and backward propagation
"""