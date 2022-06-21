import numpy as np
import torch
import torchUtils
import copy

class DQNAgent(object):

    def __init__(self,q_func, optimizer, replay_buffer, batch_size, replay_start_size,update_target_steps, n_act, gamma=0.9, e_greed=0.1):
        '''
        :param q_func: Q函数
        :param optimizer: 优化器
        :param replay_buffer: 经验回放器
        :param batch_size:批次数量
        :param replay_start_size:开始回放的次数
        :param update_target_steps: 同步参数的次数(更新目标Q)
        :param n_act:动作数量
        :param gamma: 收益衰减率
        :param e_greed: 探索与利用中的探索概率
        '''
        
        self.pred_func = q_func
        # 目标Q网络用deepcopy()实现 通过深度拷贝，
        # 我们实际上可以创建一个独立于原始数据的新对象，但包含相同的值，而不是为相同的值创建新的引用
        self.target_func = copy.deepcopy(q_func)
        self.update_target_steps = update_target_steps

        self.global_step = 0

        self.rb = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.n_act = n_act  # 动作数量
        self.gamma = gamma  # 收益衰减率
        self.epsilon = e_greed  # 探索与利用中的探索概率

    # 根据经验得到action
    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        # 这里用的是pred_func()
        Q_list = self.pred_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    # 根据探索与利用得到action
    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:  #探索
            action = np.random.choice(self.n_act)
        else: # 利用
            action = self.predict(obs)
        return action
    # 改动 pred_Vs = self.pred_func(batch_obs)和next_pred_Vs = self.target_func(batch_next_obs)
    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        # predict_Q
        pred_Vs = self.pred_func(batch_obs)
        action_onehot = torchUtils.one_hot(batch_action, self.n_act)
        predict_Q = (pred_Vs * action_onehot).sum(1)
        # target_Q 用target_fun保证target_Q 不变
        next_pred_Vs = self.target_func(batch_next_obs)
        best_V = next_pred_Vs.max(1)[0]
        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        self.optimizer.zero_grad()  # 梯度归0
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_steps == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))
        # 每多少次step更新一次target Q
        if self.global_step % self.update_target_steps==0:
            self.sync_target()
    # 同步target参数方法 zip  .parameters()为网络参数，
    def sync_target(self):
        for target_param, param in zip(self.target_func.parameters(), self.pred_func.parameters()):
            # x4.copy_(x2), 将x2的数据复制到x4,并且会
            # 修改计算图,使得反向传播自动计算梯度时,计算出x4的梯度后
            # 再继续前向计算x2的梯度. 注意,复制完成之后,两者的值的改变互不影响,
            # 因为他们并不共享内存 pytorch方法
            target_param.data.copy_(param.data)
