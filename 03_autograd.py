import torch

x = torch.randn(3, requires_grad=True)    # generates tensor with gradients
print(x)
y = x * 2
print(y)

z = y*y*2
z = z.mean()
print(z)

z.backward()        # dz/dx
print(x.grad)       # gradients in tensor x

# Here z is a scaler value.

#%% Removing stored gradients

p = torch.randn(5, requires_grad=True)
q = p+p
print(p)
print(q)
v = torch.tensor([0.11, 2, 1.2, 0.01, 1], dtype=torch.float32)      # this is a vector
# p.requires_grad_(False)       # this will remove gradients
# p.detach_()                   # removes gradient
q.backward(v)
print(p.grad)

"""
lower underscore means its a inplace operator.
"""


#%%

weights = torch.ones(4, 3, requires_grad=True)
for epoch in range(3):
    out = (weights*4).sum()
    out.backward()              # stores gradients
    print(weights.grad)
    weights.grad.zero_()        # before next operation, gradient must be cleared.
