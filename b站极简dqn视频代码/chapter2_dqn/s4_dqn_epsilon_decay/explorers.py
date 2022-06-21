import numpy as np


class EpsilonGreedy():

    def __init__( self,n_act, e_greed, decay_rate):
        self.n_act = n_act  # 动作数量
        self.epsilon = e_greed  # 探索与利用中的探索概率
        self.decay_rate = decay_rate # 衰减值

    def act(self,predct_method,obs):
        if np.random.uniform(0, 1) < self.epsilon:  #探索
            action = np.random.choice(self.n_act)
        else: # 利用
            action = predct_method(obs)
            # 𝜀的值可随着智能体与环境的交互(act)次数增多而减少 𝜀 - rate
            # max(0,x)类似于relu函数保证探索率不会衰减到0
        self.epsilon = max(0.01,self.epsilon-self.decay_rate)
        return action

