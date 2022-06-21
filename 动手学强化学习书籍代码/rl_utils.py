from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        # 当deque元素已满且有新元素从一端“入队”时，数量相同的旧元素将从另一端“出队” (被移除)
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        """*args：用来发送一个非键值对的可变数量的参数列表给一个函数
           **kwargs：是将不定长的键值对作为参数传递给一个函数
            相关的例子有很多，而且也很简单，这里不再赘述，总结一句话就是：*和**都有打包和解包的作用，
            在定义函数的时候使用，是打包的意思，在调用函数的时候则是解包的作用。
        """
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):  # on-policy没有replaybuffer,目标策略和行为策略是同一个策略
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar: # tqdm进度条可视化
            for i_episode in range(int(num_episodes/10)): 
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # on-policy没有replaybuffer,目标策略和行为策略是同一个策略
                    # 直接用交互(行为策略)得到的数据进行目标策略更新
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict) # 一个episode后更新每步网络参数,因为优势函数lmbda=1时为每一步差分
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list  # [G1,G2,...]每个episode的总奖励List

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10): # 每10个episodes输出十个episodes的总奖励
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)): # 10次迭代 ， num_episodes = 100
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size: # 这里每次交互(step)存储经验后都更新参数(learn)
                        """
                          当buffer大于多少开始学习Q和每交互多少次开始学习Q
                          if len(self.rb) > self.replay_start_size and self.global_step%self.rb.num_steps==0:
                          # *表示传入形参或zip解包
                          self.learn_batch(*self.rb.sample(self.batch_size))
                        """
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict) # 更新batch_size的参数 1/N
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy() # TD error = A(1)
    advantage_list = []
    advantage = 0.0
    # 根据train_on_policy_agent(),这里的s,a,r,s'序列有T个（一个epiode的）
    # 由上可知td_delta也会有T个，来产生TD(lmbda)的advantage，因此是td_delta[::-1]
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                