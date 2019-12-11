import gym
import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import torch.nn.functional as F
import numpy as np

env = gym.make('Taxi-v3')

n_features = 1
n_outputs = 6

memorycapacity = 100
Gamma = 0.9
EPISILON = 0.3
lr = 0.01
Batch_size = 25

class Net(nn.Module):
    def __init__(self, n_features=n_features, n_outputs=n_outputs):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(n_features, 8)
        self.layer1.weight.data.normal_(0, 1)
        self.layer2 = nn.Linear(8, n_outputs)
        self.layer2.weight.data.normal_(0, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        return self.layer2(x)

class Dqn():
    def __init__(self, memorycapacity=memorycapacity,
     Gamma=Gamma, lr=lr, EPISILON=EPISILON, Batch_size=Batch_size):
        self.Gamma = Gamma
        self.memorycapacity = np.zeros((memorycapacity, n_features*2+2))
        self.Batch_size = Batch_size
        self.EPISILON = EPISILON
        self.Batch_size = Batch_size
        self.lr = lr
        self.net = Net()
        self.action_net = Net()
        self.learncounter = 0
        self.memorycounter = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()


    def choose_action(self, state):
        state = Variable([state]).float()     # unsqueeze()增加一维，方便store
        if np.random.uniform() < min(self.EPISILON*(1.0001**self.learncounter), 1):
            action_list = list(self.action_net(state).data.numpy())
            action = action_list.index(max(action_list))
        else:
            action = np.random.choice(list(range(n_outputs)))
        return action
    
    def store_data(self, s, a, r, s_):
        data = np.hstack((s,a,r,s_))
        index = self.memorycounter % len(self.memorycapacity)
        self.memorycapacity[index] = data
        self.memorycounter += 1

    def learn(self):
        # target net update
        if self.learncounter % 100 == 0:
            self.action_net = self.net
        sample_index = np.random.choice(memorycapacity, Batch_size)
        b_memory = self.memorycapacity[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :n_features]))
        b_a = Variable(torch.LongTensor(b_memory[:, n_features: n_features+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, n_features + 1: n_features+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -n_features: ]))

        for i in range(self.Batch_size):
            q_eval = self.net(b_s[i])
            q_next = self.net(b_s_[i]).detach()
            q_target = b_r + self.Gamma* q_next.max()
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.learncounter += 1

if __name__ == "__main__":
    
    dqn = Dqn()
    print('\nCollecting experience...')
    dqn.net = torch.load('taxinet.pkl')
    for i_episode in range(1200):
        s = env.reset()
        t = 0
        reward_sum = [0 for i in range(200)]
        reward = 0
        while True:
            a = dqn.choose_action(s)
            # take action
            s_, r, done, info = env.step(a)     # if t == 200, done = true
            # modify the reward

            dqn.store_data(s, a, r, s_)
            if dqn.memorycounter > memorycapacity:
                dqn.learn()
            reward += r
            t += 1
            if r==20:
                print(f'reward is :{reward}')
                break
            s = s_
    env.close()
    torch.save(dqn.net, 'taxi_net.pkl')