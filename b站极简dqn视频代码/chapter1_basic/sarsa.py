import time

import gym
import numpy as np
import gridworld
# 乌龟远离悬崖走，胆子比较小 因为悬崖-100，根据Q更新公式越靠近Q值越低就远离了
class SarsaAgent(object):
    def __init__(self, n_status, n_act, lr=0.1, gamma=0.9, e_greed=0.1):
        self.n_status = n_status # 环境数量
        self.n_act = n_act  # 动作数量
        self.lr = lr  # 学习率
        self.gamma = gamma  # 收益衰减率
        self.epsilon = e_greed  # 探索与利用中的探索概率
        self.Q = np.zeros((n_status, n_act)) #初始化Q表格

    # 根据经验得到action greedy 
    def predict(self, state):
        #return np.argmax(self.Q[state, :]) 若有几个最大值仅会采取第一个最大的下标
        Q_list = self.Q[state, :] 
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max())) #若最大值不止一个，则随机采样
        return action

    # 根据探索与利用得到action
    def act(self, state):
        if np.random.uniform(0, 1) < self.epsilon:  #探索 e
            action = np.random.choice(self.n_act)
        else: # 利用 greedy
            action = self.predict(state)
        return action

    # 更新Q表格（Q更新公式）
    def learn(self, state, action, reward, next_state, next_action, done):
        predict_Q = self.Q[state, action] # 取Q表索引（行，列）
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.lr * (target_Q - predict_Q)  # 修正q

# 训练一轮游戏
def train_episode(env, agent,is_render):
    total_reward = 0
    state = env.reset() # 重置环境
    action = agent.act(state) # 根据算法选择一个动作

    while True:
        next_state, reward, done, info = env.step(action) # 与环境进行一个交互
        next_action = agent.act(next_state) # 探索与利用得到下一个action，不同于测试predict选action

        agent.learn(state, action, reward, next_state, next_action, done)

        action = next_action
        state = next_state

        total_reward += reward
        if is_render:env.render() # 渲染的慢一点，初始为false
        if done: break

    return total_reward

# 测试一轮游戏
def test_episode(env, agent):
    total_reward = 0
    state = env.reset()

    while True:
        action = agent.predict(state) # 因为Q表更新后选最大值动作就是最佳测试路线，就不用action = next_action
        next_state, reward, done, _ = env.step(action) # _ 其实是info 
        total_reward += reward
        state = next_state
        env.render()
        time.sleep(0.5)  
        if done:break

    return total_reward

def train(env,episodes=500,lr=0.1,gamma=0.9,e_greed=0.1):
    agent = SarsaAgent(
        n_status=env.observation_space.n,
        n_act=env.action_space.n,
        lr=lr,
        gamma=gamma,
        e_greed=e_greed)

    is_render = False
    for e in range(episodes):
        ep_reward = train_episode(env, agent,is_render)
        print('Episode %s: reward = %.1f' % (e, ep_reward))

        # 每隔50个episode渲染一下看看效果
        if e % 50 == 0:
            is_render = True
        else:
            is_render = False

    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % (test_reward))

if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")  # 0上, 1右, 2下, 3左 悬崖
    env = gridworld.CliffWalkingWapper(env)
    train(env)