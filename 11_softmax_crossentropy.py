import numpy as np
import torch

linear_preds = np.array([.2, 1.7, 9, 1, 0.1])
actual_pred = torch.tensor([2,0,1,4,3,5])
torch_preds = torch.from_numpy(linear_preds)


def softmax(x):
    res = torch.exp(x) / torch.sum(torch.exp(x), dim=0)
    return res


outputs = softmax(torch_preds)

print(f"{outputs}")


# cross entropy
def crossentropy(x, y):
    cel = -(torch.sum(x * torch.log(y)))
    return cel


print(f"cross entropy loss: {crossentropy(actual_pred, torch_preds)}")
