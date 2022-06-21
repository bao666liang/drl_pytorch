'''深度强化学习——原理、算法与PyTorch实战，代码名称：代42-A3C算法的实验结果及分析.py'''

# 导入相应的模块
import random
import gym
# import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.distributions import Normal

""" 多线程共享内存,在多线程threading中,我们可以使用global全局变量来共享某个变量(一个进程最少有一个线程),但
    多进程内存相互隔离,要用multiprocessing库中的 Value(value('i',0))(数据类型,共享变量)在各自隔离的进程中共享变量,同时为了防止多个进程同时抢占共享内存,
    使用mp.lock进程锁来解决,   (python中可以用 with 自动获取和释放锁 Lock, 其内代码锁定期间其他线程不可以干活 ,只运行一个线程)
    例如 : 同时运行20个worker,每个worker运行10次,一共会运行200次,然而由于共享内存,在第一个进程加载value值的时候
    程序却不能阻止第二个进程加载旧的值,最后的共享值(即次数)只接收到了一次值的增加,会使次数小于200 ,
    我们只需要调用multiprocessing库中的Lock(锁)就可以保证一次只能有一个进程访问这个共享变量(with lock:)
    其实Value这个包里已经包含了锁的概念,即调用get_lock() 函数就可以自动给共享变量加锁(with counter.get_lock():),
    因为这样就不需要同时调用两个包 from multiprocessing import Process, Value, (可以去掉)Lock
 """
    
# ActorCritic网络定义
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, device):
        '''构造函数，定义ActorCritic网络中各layer'''
        super().__init__()
        self.act_limit = torch.as_tensor(act_limit, dtype=torch.float32, device=device)
        self.value_layer1 = nn.Linear(obs_dim, 256)
        self.value_layer2 = nn.Linear(256, 1)
        self.policy_layer1 = nn.Linear(obs_dim, 256)
        self.mu_layer = nn.Linear(256, act_dim)
        self.sigma_layer = nn.Linear(256, act_dim)

    def forward(self, obs):
        '''前馈函数，定义ActorCritic网络向前传播的方式'''
        value = F.relu6(self.value_layer1(obs))
        value = self.value_layer2(value)
        policy = F.relu6(self.policy_layer1(obs))
        mu = torch.tanh(self.mu_layer(policy)) * self.act_limit
        sigma = F.softplus(self.sigma_layer(policy))
        return value, mu, sigma

    def select_action(self, obs):
        '''采用随机高斯策略选择动作'''
        _, mu, sigma = self.forward(obs)
        pi = Normal(mu, sigma)
        return pi.sample().cpu().numpy()

    def loss_func(self, states, actions, v_t, beta):
        '''计算损失函数'''
        # 计算值损失
        values, mu, sigma = self.forward(states)
        td = v_t - values
        value_loss = torch.squeeze(td ** 2)
        # 计算熵损失
        pi = Normal(mu, sigma)
        # 一整条episode或达到截断长度t-t(start)的和 sum()
        log_prob = pi.log_prob(actions).sum(axis=-1)
        # 连续变量信息熵 H(x)=int(积分号):f(x)logf(x)/m(x)
        entropy = pi.entropy().sum(axis=-1)
        policy_loss = -(log_prob * torch.squeeze(td.detach()) + beta * entropy)
        # 为什么求和?
        return (value_loss + policy_loss).mean()


# Worker
class Worker(mp.Process):
    def __init__(self, id, device, env_name, obs_dim, act_dim, act_limit,
                 global_network, global_optimizer,
                 gamma, beta, global_T, global_T_MAX, t_MAX,
                 global_episode, global_return_display, global_return_record, global_return_display_record):
        '''构造函数，定义Worker中各参数'''
        super().__init__()
        self.id = id # [0,1,2..]
        self.device = device
        self.env = gym.make(env_name)
        # self.env.seed(seed)
        self.local_network = ActorCritic(obs_dim, act_dim, act_limit, self.device).to(self.device)
        self.global_network = global_network
        self.global_optimizer = global_optimizer # 一个worker计算出梯度(一个episode或达到截断长度t-t(start)的累积梯度)后传到全局网络优化更新参数
        self.gamma, self.beta = gamma, beta
        # global是所有worker总的step,t是一个worker的step
        self.global_T, self.global_T_MAX, self.t_MAX = global_T, global_T_MAX, t_MAX
        self.global_episode = global_episode
        self.global_return_display = global_return_display
        self.global_return_record = global_return_record
        self.global_return_display_record = global_return_display_record
    # 加's'表示一个episode或达到截断长度的多个
    def update_global(self, states, actions, rewards, next_states, done, gamma, beta, optimizer):
        '''更新一次全局梯度信息'''
        # 按书的流程，计算优势函数
        if done:
            R = 0
        else:  # R = V' 不是书上的G
            R, mu, sigma = self.local_network.forward(next_states[-1])
        # 一个episode或达到截断长度的奖励个数
        length = rewards.size()[0]

        # 计算真实值即目标值：R+r*V'(是估计值)   比如s1,s2,s3->r1,r2 没有r3因为s3是next_state
        v_t = torch.zeros([length, 1], dtype=torch.float32, device=self.device)
        for i in range(length, 0, -1): # A = E(R + r*V' - v)
            R = rewards[i - 1] + gamma * R
            v_t[i - 1] = R # 目标值
        #在worker网络计算损失函数，不同于A2C在global网络计算loss(每个step的平均，防止loss过大))
        loss = self.local_network.loss_func(states, actions, v_t, beta)  
        # 使用异步并行的工作组进行梯度累积，对全局网络进行异步更新
        optimizer.zero_grad()
        # 求梯度
        loss.backward()
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            # 将worker网络的每个参数的梯度赋给全局网络
            global_params._grad = local_params._grad
        # 用梯度进行；全局网络参数更新 因为optimizer = Adam(global_network.parameters(), lr=lr)
        optimizer.step()
        # 全局网络更新后将更新后的网络参数赋给worker   
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):
        '''运行游戏情节'''
        # 初始化环境和情节奖赏
        t = 0
        state, done = self.env.reset(), False
        episode_return = 0

        # 获取一个buffer的数据
        while self.global_T.value <= self.global_T_MAX:
            t_start = t
            # 每次开始前buff都要清零，因为用于更新用完一次就扔了
            buffer_states, buffer_actions, buffer_rewards, buffer_next_states = [], [], [], []
            while not done and t - t_start != self.t_MAX:
                action = self.local_network.select_action(torch.as_tensor(state, dtype=torch.float32, device=self.device))
                next_state, reward, done, _ = self.env.step(action)
                episode_return += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_next_states.append(next_state)
                buffer_rewards.append(reward / 10)
                t += 1
                # 为了解决上述不同进程抢共享资源的问题，我们可以用加进程锁来解决
                # t是每个worker里的step , global_T是所有worker加起来的总step(因此要用with lock和value)
                with self.global_T.get_lock():
                    # global的都要用.value   global_T = mp.Value('i', 0)
                    self.global_T.value += 1
                state = next_state

            # 根据这一buffer(一个episode或截断时间的数据即一个worker完成一个job的)的数据来更新全局网络梯度信息
            self.update_global(
                torch.as_tensor(buffer_states, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_actions, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_rewards, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_next_states, dtype=torch.float32, device=self.device),
                done, self.gamma, self.beta, self.global_optimizer
            )

            # 处理一次情节完成时的打印输出全局奖励操作
            if done:
                # global_episode上锁，处理情节完成时的操作
                with self.global_episode.get_lock():
                    # 只要有一个worker完成一次job，global_episode就加一 
                    self.global_episode.value += 1    
                    # global_return_record存放每次job的总奖励     global_return_record = mp.Manager().list()
                    self.global_return_record.append(episode_return)
                    if self.global_episode.value == 1:
                        self.global_return_display.value = episode_return  # 第一次完成情节
                    else:
                        self.global_return_display.value *= 0.99
                        # 当前episode总奖励占的比例少是为了看总的
                        self.global_return_display.value += 0.01 * episode_return
                        self.global_return_display_record.append(self.global_return_display.value)
                        # 10个情节输出一次
                        if self.global_episode.value % 10 == 0:
                            print('Process: ', self.id, '\tepisode: ', self.global_episode.value, '\tepisode_return: ', self.global_return_display.value)
                # 因为episode_return += reward
                episode_return = 0
                state, done = self.env.reset(), False

if __name__ == "__main__":

    # 定义实验参数
    # ##获取默认设备的Id torch.cuda.current_device() # 0
    device = 'cuda:1'
    env_name = 'Pendulum-v0' # 钟摆倒立
    num_processes = 8
    gamma = 0.9
    beta = 0.01
    lr = 1e-4
    T_MAX = 1000000 # 所有worker总和step
    t_MAX = 5 # 一个worker的job的最长step(5有点少？)

    # 设置进程启动方式为spawn
    mp.set_start_method('spawn')

    # 定义环境和进程相关参数
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high
    global_network = ActorCritic(obs_dim, act_dim, act_limit, device).to(device)
    # 相当于lambda t: t.share_memory_()  使Tensor在所有进程之间共享而不被复制
    global_network.share_memory()
    # 因为在global_model更新参数
    optimizer = Adam(global_network.parameters(), lr=lr)
    # 在多线程threading中，我们可以使用global全局变量来共享某个变量，但是在多进程中这是行不通的，
    # 我们需要用到共享内存shared memory的形式 
    # 两种共享内存的形式，一个是Value，一个是Array（其实就是list） i/d表示数据格式int/double
    global_episode = mp.Value('i', 0)
    # 在多进程中，由于进程之间内存相互是隔离的，所以无法在多个进程中用直接读取的方式共享变量，
    # 这时候就可以用multiprocessing库中的 Value在各自隔离的进程中共享变量
    global_T = mp.Value('i', 0)
    global_return_display = mp.Value('d', 0)

    """ 数据共享进程的实现,该类进程可以通过Manager类创建,
        主要支持有两类操作数据形式:(list,dict)
    """
    # 多进程写入列表 便于绘图
    global_return_record = mp.Manager().list()  #创建共享数据对象
    global_return_display_record = mp.Manager().list()

    # 定义worker 各worker进程依次开始训练(谁先训练完谁去更新)    \续行，如果一行太长了想分行，在第一行后加空格\
    workers = [Worker(i, device, env_name, obs_dim, act_dim, act_limit, \
                      global_network, optimizer, gamma, beta,
                      global_T, T_MAX, t_MAX, global_episode,
                      global_return_display, global_return_record, global_return_display_record) for i in range(num_processes)]

    [worker.start() for worker in workers] # 启动每一个worker
    # 主线程创建并启动子线程，如果自线程中要进行大量的耗时运算，主线程往往将早于子线程结束之前结束。如果主线程想等待子线程执行完成之后再结束，
    # 比如子线程处理一个数据，主线程要取得这个数据中的值，就要用到 join() 方法
    [worker.join() for worker in workers]  # 主进程等待调用join的子进程结束再往下运行(因为worker中一开始有global_T_max限制，所以可以循环写) 

    # 保存模型
    torch.save(global_network, 'a3c_model.pth')
    
    # 实验结果可视化
    import matplotlib.pyplot as plt
    save_name = 'a3c_gamma=' + str(gamma) + '_beta=' + str(beta) + '_' + env_name
    save_data = np.array(global_return_record)
    plt.plot(np.array(global_return_display_record))
    np.save(save_name + '.npy', save_data)
    plt.ylabel('return')
    plt.xlabel('episode')
    plt.savefig(save_name + '.png')