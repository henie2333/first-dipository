import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

#定义最高次项系数
n = 3

def make_features(x):
    """构建一个[x,x^2,x^3]矩阵特征的实例i.e"""
    # unqueeze(1)为增加一维
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,n+1)],1)

# y = 1.7 + 2.4x + 0.3x^2 + 1.1x^3
W_target = torch.FloatTensor([2.4, 0.3, 1.1]).unsqueeze(1)
b_target = torch.FloatTensor([1.7])

def f(x):
    """近似功能"""
    return x.mm(W_target)+b_target[0]

def get_batch(batch_size=32, random=None):
    if random is None:
        random = torch.randn(batch_size)
    batch_size = random.size()[0]
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

# 定义模型
class ploy_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.poly = nn.Linear(n, 1)     # 任然为一层，输出只有y_predict值，仍为1

    def forward(self, x):
        out = self.poly(x)
        return out

model = ploy_model()

# 损失函数和优化器
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

epoch = 0
while True:
    # 获得数据
    batch_x, batch_y = get_batch()
    # 向前传播
    output = model(batch_x)
    loss = loss_func(output, batch_y)
    print_loss = loss.data.item()
    # 重置梯度
    optimizer.zero_grad()
    # 后向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    epoch +=1
    if print_loss < 1e-3:
        break

print("the number of epoches :", epoch)

# 定义函数输出形式
def func_format(weight, bias, n):
    func = ''
    for i in range(n, 0, -1):
        func += ' {:.2f} * x^{} +'.format(weight[i - 1], i)
    return 'y =' + func + ' {:.2f}'.format(bias[0])

predict_weight = model.poly.weight.data.numpy().flatten()
predict_bias = model.poly.bias.data.numpy().flatten()
print('predicted function :', func_format(predict_weight, predict_bias, 3))
real_W = W_target.numpy().flatten()
real_b = b_target.numpy().flatten()
print('real      function :', func_format(real_W, real_b, 3))
