'''深度强化学习——原理、算法与PyTorch实战，代码名称：代43-A2C算法的实验结果及分析.py'''

# 导入相应的模块
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp  # 多进程
from torch.optim import Adam
from torch.distributions import Normal

"""1.一个在内存中运行的应用程序。每个进程都有自己独立的一块内存空间
   2.一个进程可以有多个线程,比如在Windows系统中,一个运行的xx.exe就是一个进程
   进程中的一个执行任务（控制单元）,负责当前进程中程序的执行。一个进程至少有一个线程,
   一个进程可以运行多个线程,多个线程可共享数据,一个线程崩溃整个进程都死掉
"""

# ActorCritic网络定义
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, device):
        '''构造函数，定义ActorCritic网络中各layer'''
        super().__init__()
        """as_tensor方法和from_array方法使用的数据和numpy array使用的数据是一样的，
           也就是说它们和numpy array其实是共享内存的。而Tensor方法和tensor方法在转换的时候，则是开辟了新的内存空间来存储tensor的
           浅拷贝和深拷贝的区别，前者效率高，但改变tensor值会使numpy值也会改变，要注意
           如果我们再转换numpy array时采用的是from_array和as_tensor就会导致numpy array里面的值也被修改
        """
        self.act_limit = torch.as_tensor(act_limit, dtype=torch.float32, device=device)
        self.value_layer1 = nn.Linear(obs_dim, 256) # 用的优势函数 A = Q - V = E[R+rV'-V] 约等 R+rV'-V 因此只需要V(s)不需要Q(s,a)
        self.value_layer2 = nn.Linear(256, 1)
        self.policy_layer1 = nn.Linear(obs_dim, 256) 
        self.mu_layer = nn.Linear(256, act_dim) # 输出每个动作维度的均值
        self.sigma_layer = nn.Linear(256, act_dim)

    def forward(self, obs):
        '''前馈函数，定义ActorCritic网络向前传播的方式'''
        # ReLU6就是普通的ReLU但是限制最大输出值为6（对输出值做clip）
        value = F.relu6(self.value_layer1(obs))
        value = self.value_layer2(value)
        policy = F.relu6(self.policy_layer1(obs))
        mu = torch.tanh(self.mu_layer(policy)) * self.act_limit # 因为tanh后(-1,1)*act_limit
        # Softplus函数可以看作是ReLU函数的平滑 为什么用relu输出方差 ？
        sigma = F.softplus(self.sigma_layer(policy))
        return value, mu, sigma

    def select_action(self, obs):
        '''采用随机高斯策略选择连续动作'''
        _, mu, sigma = self.forward(obs)
        pi = Normal(mu, sigma)
        # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
        return pi.sample().cpu().numpy()

    def loss_func(self, states, actions, v_t, beta):
        '''计算损失函数'''
        # 计算值损失
        values, mu, sigma = self.forward(states)
        td = v_t - values
        #squeeze()函数的功能是维度压缩。返回一个tensor（张量），其中 input 中大小为1的所有维都已删除。
        # [[]] -> [] 
        value_loss = torch.squeeze(td ** 2)
        # 计算熵损失
        pi = Normal(mu, sigma)
        log_prob = pi.log_prob(actions).sum(axis=-1)
        entropy = pi.entropy().sum(axis=-1)
        policy_loss = -(log_prob * torch.squeeze(td.detach()) + beta * entropy)
        return (value_loss + policy_loss).mean()


#  mp.Process：由该类实例化得到的对象，表示一个子进程中的任务（尚未启动）
#  方法 ： p.start()：启动进程，并调用该子进程中的p.run()  
#  p.join([timeout])：主线程等待p终止（强调：是主线程处于等的状态，而p是处于运行的状态）
#  timeout是可选的超时时间，需要强调的是，p.join只能join住start开启的进程，而不能join住run开启的进程
class Worker(mp.Process):
    def __init__(self, id, device, env_name, global_network_lock,  # obs_dim, act_dim, act_limit,
                 global_network, global_optimizer,
                 gamma, beta, global_T, global_T_MAX, t_MAX,
                 global_episode, global_return_display, global_return_record, global_return_display_record):
        '''构造函数，定义Worker中各参数'''
        super().__init__()
        self.id = id
        self.device = device
        self.env = gym.make(env_name)
        self.global_network_lock = global_network_lock
        self.global_network = global_network
        self.global_optimizer = global_optimizer
        self.gamma, self.beta = gamma, beta
        self.global_T, self.global_T_MAX, self.t_MAX = global_T, global_T_MAX, t_MAX
        self.global_episode = global_episode
        self.global_return_display = global_return_display
        self.global_return_record = global_return_record
        self.global_return_display_record = global_return_display_record

    def update_global(self, states, actions, rewards, next_states, done, gamma, beta, optimizer):
        '''更新一次全局梯度信息'''
        if done:
            R = 0
        else:
            R, mu, sigma = self.global_network.forward(next_states[-1])
        length = rewards.size()[0]

        # 计算真实值
        v_t = torch.zeros([length, 1], dtype=torch.float32, device=self.device)
        for i in range(length, 0, -1):
            R = rewards[i - 1] + gamma * R
            v_t[i - 1] = R
        #在全局网络中计算损失函数 worker仅采样数据，不同于A3C
        loss = self.global_network.loss_func(states, actions, v_t, beta)  
        # A2C的工作组不用于累计梯度，只在采样总量到达batch大小时更新参数  
        # 两个进程都会把value拷贝到自己的私有内存然后进行处理，并写回到共享值里 
        # t_max或done的step就是batch_size,一个worker运行完就更新 ？parl是actor.py作为workers进行sample batch_size后进行更新
        with self.global_network_lock.get_lock():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def run(self):
        '''运行游戏情节'''
        # 初始化环境和情节奖赏
        t = 0
        state, done = self.env.reset(), False
        episode_return = 0

        # 获取一个buffer的数据
        while self.global_T.value <= self.global_T_MAX:
            t_start = t
            buffer_states, buffer_actions, buffer_rewards, buffer_next_states = [], [], [], []
            while not done and t - t_start != self.t_MAX:
                action = self.global_network.select_action(torch.as_tensor(state, dtype=torch.float32, device=self.device))
                next_state, reward, done, _ = self.env.step(action)
                episode_return += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_next_states.append(next_state)
                buffer_rewards.append(reward / 10)
                t += 1
                # 两个worker同时完成episode后同时访问global_T,但get_lock一次只允许一个进程访问共享变量
                with self.global_T.get_lock():
                    self.global_T.value += 1
                state = next_state
            # 根据这一buffer的数据来更新全局梯度信息
            self.update_global(
                torch.as_tensor(buffer_states, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_actions, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_rewards, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_next_states, dtype=torch.float32, device=self.device),
                done, self.gamma, self.beta, self.global_optimizer
            )

            # 处理情节完成时的操作
            if done:
                # global_episode上锁，处理情节完成时的操作
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
                    self.global_return_record.append(episode_return)
                    if self.global_episode.value == 1:
                        self.global_return_display.value = episode_return  # 第一次完成情节
                    else:
                        self.global_return_display.value *= 0.99
                        self.global_return_display.value += 0.01 * episode_return
                        self.global_return_display_record.append(self.global_return_display.value)
                        # 10个情节输出一次
                        if self.global_episode.value % 10 == 0:
                            print('Process: ', self.id, '\tepisode: ', self.global_episode.value, '\tepisode_return: ', self.global_return_display.value)

                episode_return = 0
                state, done = self.env.reset(), False


if __name__ == "__main__":

    # 定义实验参数
    device = 'cuda:1'
    env_name = 'Pendulum-v0'
    num_processes = 8
    gamma = 0.9
    beta = 0.01
    lr = 1e-4
    T_MAX = 1000000
    t_MAX = 5

    # 设置进程启动方式为spawn
    # spawn：使用此方式启动的进程，只会执行和 target 参数或者 run() 方法相关的代码
    # Windows 平台只能使用此方法,效率最低
    mp.set_start_method('spawn')

    # 定义环境和进程相关参数
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high
    global_network_lock = mp.Value('i', 0)
    global_network = ActorCritic(obs_dim, act_dim, act_limit, device).to(device)
    global_network.share_memory()
    optimizer = Adam(global_network.parameters(), lr=lr)
    global_episode = mp.Value('i', 0)
    global_T = mp.Value('i', 0)
    global_return_display = mp.Value('d', 0)
    global_return_record = mp.Manager().list()
    global_return_display_record = mp.Manager().list()

    # 定义workers =  [worker0,worker1,...]    各worker进程开始训练
    workers = [Worker(i, device, env_name, global_network_lock,  # obs_dim, act_dim, act_limit, \
                      global_network, optimizer, gamma, beta,
                      global_T, T_MAX, t_MAX, global_episode,
                      global_return_display, global_return_record, global_return_display_record) \
               for i in range(num_processes)]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]


    # 保存模型
    torch.save(global_network, 'a2c_Pendulum-v0_model.pth')

    # 实验结果可视化
    import matplotlib.pyplot as plt
    save_name = 'a2c_gamma=' + str(gamma) + '_beta=' + str(beta) + '_' + env_name
    save_data = np.array(global_return_record)
    plt.plot(np.array(global_return_display_record))
    np.save(save_name + '.npy', save_data)
    plt.ylabel('return')
    plt.xlabel('episode')
    plt.savefig(save_name + '.png')