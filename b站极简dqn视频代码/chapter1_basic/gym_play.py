import gym
import random
import gridworld 
# 悬崖行走
env = gym.make("CliffWalking-v0") # 声明一个环境 CartPole-v0 平衡杆
env = gridworld.CliffWalkingWapper(env)
#CartPole-v1
state = env.reset() # 重置环境，返回最初的状态
while True:
    #action = random.randint(0,3) 四个动作1，2，3，4，两种写法
    action = env.action_space.sample()
    state, reward, done, info = env.step(action) 
    # 与环境交互，done表示是否完成（黄格终点或黑格即悬崖-100）白格-1

    env.render() #渲染一帧动画
    if done:  # 到达一次就结束
        break