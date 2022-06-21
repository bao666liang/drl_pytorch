import random
import collections
from torch import FloatTensor

class ReplayBuffer(object):
    def __init__(self, max_size, num_steps=3 ):
        self.buffer = collections.deque(maxlen=max_size)
        self.num_steps  = num_steps
    # 向buffer添加样本经验
    def append(self, exp):
        self.buffer.append(exp)
    # 从buffer提取样本经验
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # 训练时要变成张量形式
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        next_obs_batch = FloatTensor(next_obs_batch)
        done_batch = FloatTensor(done_batch)
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch

    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    # deque的好处是maxlen=3是取三个，a第一个（1，1）就没有了，取后三个
    a=collections.deque(maxlen=3)
    print(a)
    a.append((1,1))
    a.append((2,2))
    a.append((3,3))
    a.append((4,4))
    print(a)
    # zip的作用是将[（1，1），（2，2）]变为[(1,2),(1,2)]即状态和动作分开
    state, action = zip(*a)
    print(state, action)
