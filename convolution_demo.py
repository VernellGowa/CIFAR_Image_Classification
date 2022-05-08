import numpy as np
import torch as T
device = T.device('cpu')

# -----------------------------------------------------------

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 2, 2)  # chnl-in, out, krnl
        

    def forward(self, x):
        z = self.conv1(x)
        return z

# -----------------------------------------------------------


net = Net().to(device)

x = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]])

x = x.reshape(1, 1, 4, 4)  # bs, chnls, rows, cols
x = T.tensor(x, dtype=T.float32).to(device)


z = net(x)
print(z.shape)
print(z)

print("\nEnd convolution demo ")