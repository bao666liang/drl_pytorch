#深度强化学习——原理、算法与PyTorch实战，代码名称：代41-TD3算法的实验过程.py
import numpy as np
import torch
import gym
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size # 最大容量
        self.ptr = 0 # 添加索引
        self.size = 0 # 容量

        self.state = np.zeros((max_size, state_dim)) # 一行一个状态
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size # 每添加一次索引加一直到max时归零
        self.size = min(self.size + 1, self.max_size) # 不超过max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size) # 采样索引

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        # 全连接层需要我们把输入拉直成一个列向量 不同于卷积神经网络[,,,]
        # 输入是图像需要进行标准化（x-u）/std 神经网络学习的本质就是为了学习的数据分布，
        # 一旦训练数据与测试数据分布不同，那么网络的泛化性能会大大降低 
        # 标准化的作用仅仅是将数据拉回到同一个量级，使网络更容易学习
class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(Actor, self).__init__()

            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, action_dim)

            self.max_action = max_action # 改为倒立摆后 state[,,]  

        def forward(self, state):
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            return self.max_action * torch.tanh(self.l3(a)) # 输出为连续动作值[-2,2]
           
            

class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Critic, self).__init__()
            # 为解决过高估，用较小的Q值去构造Critic学习的Target Value
            # Q1 architecture
            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)

            # Q2 architecture
            self.l4 = nn.Linear(state_dim + action_dim, 256)
            self.l5 = nn.Linear(256, 256)
            self.l6 = nn.Linear(256, 1)

        def forward(self, state, action):
            # 前向传播时求Q(s,a) 需要拼接再输入 行向量？
            sa = torch.cat([state, action], 1)

            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = F.relu(self.l4(sa))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2
        # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        def Q1(self, state, action):
            sa = torch.cat([state, action], 1)

            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1

# actor1=Actor(17,6,1.0)
actor1=Actor(3,1,2.0)
# self.children()只包括网络模块的第一代儿子模块，而self.modules()包含网络模块的自己本身和所有后代模块。
for ch in actor1.children():
    print(ch)
print("*********************")
critic1=Critic(3,1)
for ch in critic1.children():
    print(ch)

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor) # 创建目标策略网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic) # 创建目标价值网络
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        # 化为向量形式方面linear网络输入
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1 # 用于策略网络和价值网络的不同步

        # Sample replay buffer 都应该加s（多数）
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        """ 在交互的时候,a加上了了noise。这个时候的noise是为了更充分地开发整个游戏空间。
            计算target_Q的时候,a'加上noise,是为了预估更准确,网络更有健壮性。
            更新actor网络的时候,我们不需要加上noise,这里是希望actor能够寻着最大值。加上noise并没有任何意义
        """
        # pytorch对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，
        # 可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建，反向传播
        # 因为后面要取最小值和软更新 不是立即改变网络参数
        with torch.no_grad():
            # Select action according to policy(探索) noise and add clipped noise
            # randn_like返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            # clip(a' + noise）-> (-2.0,2.0) clip防止action超出范围
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) # 用target网络的Double Q'，一共6个网络
            target_Q = reward + not_done * self.discount * target_Q # y

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss （Double Q的loss和）
        # torch.mean()有必要加上   梯度累积时，记得要除以累积次数 ，不然梯度会太大导致训练异常(actor和critic) 这里加不加效果一样
        critic_loss = torch.mean(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse 评论家更新d次后行动家再更新 mean():batch_size 负号：最大化J
            # 更新行动家网络时的a不需要加noise,因为这只是 Loss = -Q(s,a) 即最大化Q（J）
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models 软更新
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        #  pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系
        #  torch.save(model.state_dict(), PATH) 
        #  常用的保存state_dict的格式是".pt"或'.pth'的文件,即下面命令的 PATH="./***.pt"
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment 评估策略
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100) #固定随机种子生成的随机值保证每次训练相同

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False # done = False
        while not done:
            # env.render()
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            

    avg_reward /= eval_episodes # 平均每个episode的奖励

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


policy = "TD3"
env_name = "Pendulum-v0"  # OpenAI gym environment name
seed = 0  # Sets Gym, PyTorch and Numpy seeds
start_timesteps = 25e3  # 如果小于25e3步，就随机，如果大于就用get-action ：e-greedy策略
eval_freq = 5e3  # How often (time steps) we evaluate 5e3
max_timesteps = 1e6  # Max time steps to run environment 10e6
expl_noise = 0.1  # Std of Gaussian exploration noise
batch_size = 256  # Batch size for both actor and critic
discount = 0.99  # Discount factor
tau = 0.005  # Target network update rate
policy_noise = 0.2  # Noise added to target policy during critic update
noise_clip = 0.5  # Range to clip target policy noise
policy_freq = 2  # Frequency of delayed policy updates
save_model = "store_true"  # Save model and optimizer parameters
load_model = ""  # Model load file name, "" doesn't load, "default" uses file_name

# Python3.6新加特性，前缀f用来格式化字符串。可以看出f前缀可以更方便的格式化字符串,比format()方法可读性高且使用方便
file_name = f"{policy}_{env_name}_{seed}"
print("---------------------------------------")
print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if save_model and not os.path.exists("./models"):
    os.makedirs("./models")

env = gym.make(env_name)

# Set seeds :env pytorch numpy
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": discount,
    "tau": tau,
    "policy_noise": policy_noise * max_action,
    "noise_clip": noise_clip * max_action,
    "policy_freq": policy_freq
}
# **表示有键值对的可变形参
policy = TD3(**kwargs)

if load_model != "":
    policy_file = file_name if load_model == "default" else load_model
    policy.load(f"./models/{policy_file}")

replay_buffer = ReplayBuffer(state_dim, action_dim)

# 初始化网络 此处需要额外调用以使内部函数能够使用 model.forward
evaluations = [eval_policy(policy, env_name, seed)]

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
# take_action- > step-> collect -> Train -> evaluate -> save model and result
for t in range(int(max_timesteps)):

    episode_timesteps += 1
    

    
    # 如果小于25000步就只交互进行存储，大于后开始训练，训练动作用 a+noise代替e-greedy(这是DQN)（同ddpg）
    if t < start_timesteps: 
        action = env.action_space.sample()
    else:
        action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
        ).clip(-max_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
    # env._max_episode_steps = 200 是强制改变最大step防止done
    
     

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= start_timesteps:
        policy.train(replay_buffer, batch_size)

    if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True 
        print(
            f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        episode_reward = 0 # 一个episode的G
        episode_timesteps = 0 # 一个episode所经过的step
        episode_num += 1 # episode数

        # Evaluate episode 这里的动作不加noise
    if (t + 1) % eval_freq == 0:
        evaluations.append(eval_policy(policy, env_name, seed))
        np.save(f"./results/{file_name}", evaluations)

    if save_model:
        policy.save(f"./models/{file_name}")

state_dim