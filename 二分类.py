import torch as t
import torch.autograd.variable as variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as plot


p1=[[1.24,1.27],[1.36,1.74],[1.38,1.64],[1.38,1.82],[1.38,1.90],[1.40,1.70],[1.48,1.82],[1.54,1.82],[1.56,2.08]]
p2=[[1.14,1.82],[1.18,1.96],[1.20,1.86],[1.26,2.00],[1.28,2.00],[1.30,1.96]]
p = p1[:-2]+p2[:-2]
target = []
for each in p:
    if each in p1:target.append([1, 0])
    elif each in p2:target.append([0,1])
target = variable(t.tensor(np.array(target)).type(t.FloatTensor))
p = variable(t.tensor(np.array(p)).type(t.FloatTensor))


class net(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(net, self).__init__()
        self.hidden = nn.Linear(n_features, 8)
        self.output = nn.Linear(8, n_outputs)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return F.softmax(x)
        
Net = net(2,2)
predict = Net(p)

opitmizer = t.optim.SGD(Net.parameters(),lr=0.03)
loss_fun = nn.MSELoss()   #选择 均方差为误差函数

 
for i in range(3000):       # 训练过程
    Length = len(p)
    predict = Net(p)
    loss = loss_fun(predict, target)
    opitmizer.zero_grad()
    loss.backward()
    opitmizer.step()

print(Net(variable(t.tensor(np.array(p1[-2:]+p2[-2:])).type(t.FloatTensor))))       # 测试过程