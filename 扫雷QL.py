"""
失败的扫雷drl尝试
"""
import numpy as np
import torch as t
import torch.nn as nn
import torch.autograd.variable as Variable


class Model():
    def __init__(self, row, col):
        self.width = col
        self.height = row
        self.items = [[0 for c in range(col)] for r in range(row)]
        self.initmine()
        self.end_flag = False
 
    def setItemValue(self, r, c, value):
        """
        设置某个位置的值为value
        """
        self.items[r][c]=value;
 
    def checkValue(self, r, c, value=0):
        """
        检测某个位置的值是否为value
        """
        if self.items[r][c]!=-1 and self.items[r][c]==value:
            self.items[r][c]=-1 #已经检测过
            return True
        else:
            return False
     
    def countValue(self, r, c):
        """
        统计某个位置周围8个位置中，值为value的个数  #1代表雷
        """
        count=0
        if r-1 >= 0 and c-1 >= 0:
            if self.items[r-1][c-1]==1:
                count += 1
        if r-1 >= 0 and c>=0:
            if self.items[r-1][c] == 1:
                count += 1
        if r-1 >= 0 and c+1 <= self.width-1:
            if self.items[r-1][c+1] == 1:
                count += 1
        if c-1 >= 0:
            if self.items[r][c-1] == 1:
                count += 1
        if c+1<=self.width-1:
            if self.items[r][c+1] == 1:
                count += 1
        if r+1 <= self.height-1 and c-1 >= 0:
            if self.items[r+1][c-1]==1:
                count += 1
        if r+1 <= self.height-1:
            if self.items[r+1][c]==1:
                count += 1
        if r+1 <= self.height-1 and c+1 <= self.width-1:
            if self.items[r+1][c+1] == 1:
                count += 1
        self.items[r][c] = count
    
    def initmine(self):
        """
        埋雷,每行埋height/width+2个暂定
        """
        r=np.random.randint(1, self.height//self.width+2)
        for r in range(self.height):
            for i in range(2):
                c = np.random.choice(list(range(self.width)))
                self.setItemValue(r, c, 1)
    
    def fresh(self):
        for each in self.items:
            for a in each:
                print(a, end=' ')
            print()
        return self.items
    
    def is_win(self):
        for each in self.items:
            if 0 in each:
                return False
        return True

    def get_state(self):
        return np.array(self.items, dtype='float32')
    
    def take_action(self, action):      # action=pos
        r = action//self.width
        c = action%self.height
        if not self.checkValue(r,c):
            reward = -50
            self.end_flag = True
        else:
            self.countValue(r, c)
            if self.is_win():
                reward = 100
                self.end_flag = True
            else:
                reward = 5
        return reward

        

class Net(nn.Module):
    def __init__(self, row, col):
        super(Net, self).__init__()
        self.linear1 = t.nn.Linear(row*col, 40)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(40, row*col)
        self.criterion = t.nn.MSELoss()
        self.opt = t.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        return y


Gamma = 0.98
lr = 0.03
EPISILON = 0.9
row = 5
col = 5
n_outputs = row*col


class Dqn():
    def __init__(self, Gamma=Gamma, lr=lr, EPISILON=EPISILON, row=row, col=col):
        self.Gamma = Gamma
        self.EPISILON = EPISILON
        self.lr = lr
        self.action_net = Net(row, col)
        self.optimizer = t.optim.Adam(self.action_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = Variable(np.array(state)).float()    
        if np.random.uniform() < self.EPISILON:
            action_list = self.action_net(state).data.numpy()
            action = int(np.where(action_list==np.max(action_list))[0])
        else:
            action = np.random.choice(list(range(n_outputs)))
        return action
    
    def learn(self, s, r, s_):
        s = Variable(np.array(s, dtype='float32'))
        s_ = Variable(np.array(s_, dtype='float32'))
        q_eval = self.action_net(s)
        q_next = self.action_net(s_)
        q_target = r + self.Gamma * q_next.max()
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def transmit(state):
    tmp = []
    for each in state:
        tmp += each
    return tmp


if __name__ == "__main__":
    dqn = Dqn()
    graph = Model(row, col)
    e = 0
    while e < 100:
        total_reward = 0
        step = 0
        while not graph.end_flag:
            if step >= 200:
                break
            state = transmit(graph.items)
            action = dqn.choose_action(state)
            reward = graph.take_action(action)
            total_reward += reward
            next_state = transmit(graph.items)
            dqn.learn(state, reward, next_state)
            step += 1
        graph = Model(row, col)
        if total_reward >= 0:
            print('total_reward={}, total_step={}'.format(total_reward, step))
            e += 1





