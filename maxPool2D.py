import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print("\nBegin PyTorch max pooling demo ")

x = np.arange(16, dtype=np.float32)
x = x.reshape(1, 1, 4, 4)  # bs, channels, height, width
X = torch.tensor(x, dtype=torch.float32).to(device)

xx = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

print("\nSource input: ")
print(X)

pool1 = nn.MaxPool2d(2, stride=1)
z1 = pool1(X)
print("\nMaxPool with kernel=2, stride=1: ")
print(z1)

pool2 = nn.MaxPool2d(2, stride=2)
z2 = pool2(X)
print("\nMaxPool with kernel=2, stride=2: ")
print(z2)

print("\nEnd max pooling demo ")