import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as variable
import matplotlib.pyplot as plt


x = [5, 20, 1, 8, 2, 9, 4, 22, 4, 4, 8, 6, 8, 4, 0, 5, 0, 49, 6, 1, 37, 9, 4]
y = [26, 70, 56, 24, 13, 13, 6, 41, 33, 3, 6, 13, 13, 16, 3, 13, 0, 80, 12, 13, 44, 16, 6]
y = t.tensor(np.array(y)).type(t.FloatTensor)       
x = t.tensor(np.array(x)).type(t.FloatTensor)
x = t.unsqueeze(x, dim=1)           # 转置为23*1的列向量，便于计算
y = t.unsqueeze(y, dim=1)
x = variable(x)
y = variable(y)

class linear_net(nn.Module):
    def __init__(self, n_features=1, n_outputs=1):
        super(linear_net, self).__init__()
        self.output = nn.Linear(n_features, n_outputs)  # 线性回归只需要一层

    def forward(self, x):
        return self.output(x)


net = linear_net()
optimizer = t.optim.SGD(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

for e in range(500):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.scatter(x.data.numpy(), y.data.numpy())
out = out.data.numpy()
x = x.data.numpy()
k = (out[-1]-out[0])/(x[-1]-x[0])
b = out[0]
plt.plot(x, out)
plt.title('regression:y={}x+{}'.format(k, b))
plt.show()
