#!/usr/bin/env python3


import rospy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from collections import namedtuple
from std_msgs.msg import Float32MultiArray
from dqn.env.env_dqn_LIDAR_PER import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

EPISODES = 1001



class DQN(nn.Module):
    def __init__(self, state_size, action_size): 
        super(DQN, self).__init__() 
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.drp1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(200, self.action_size)  
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drp1(x)
        x = self.fc3(x)
        return x
        
  
class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)





# TD오차를 저장할 메모리 클래스

TD_ERROR_EPSILON = 0.0001  # 오차에 더해줄 바이어스


class TDerrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 메모리의 최대 저장 건수
        self.memory = []  # 실제 TD오차를 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수

    def push(self, td_error):
        '''TD 오차를 메모리에 저장'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 메모리가 가득차지 않은 경우

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity  # 다음 저장할 위치를 한 자리 뒤로 수정

    def __len__(self):
        '''len 함수로 현재 저장된 갯수를 반환'''
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''TD 오차에 따른 확률로 인덱스를 추출'''

        # TD 오차의 합을 계산
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 충분히 작은 값을 더해줌

        # batch_size 개만큼 난수를 생성하고 오름차순으로 정렬
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 위에서 만든 난수로 인덱스를 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # TD_ERROR_EPSILON을 더한 영향으로 인덱스가 실제 갯수를 초과했을 경우를 위한 보정
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        '''TD 오차를 업데이트'''
        self.memory = updated_td_errors




   


class Brain():
    def __init__(self, state_size, action_size):
        #self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('src/dqn', 'save_model/0729_dqn_lidar_')
        self.result = Float32MultiArray()
        
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 4000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 500
        self.train_start = 1000
        self.CAPACITY = 100000
        self.memory = ReplayMemory(self.CAPACITY)       

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        
        print(self.model)
        
        self.loss = 0.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        
        self.td_error_memory = TDerrorMemory(self.CAPACITY)
        
        
        if self.load_model:
            self.model.load(self.dirPath+str(self.load_episode)+".pt")

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                
    def decide_action(self, state, episode):
    
        if self.epsilon <= np.random.rand():
            #print("모델에 의한 행동선택")
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
                #print("action : ", action.item())
        else:
            action = torch.LongTensor([[random.randrange(self.action_size)]]).to(device)
            #print("무작위 행동선택 action : ", action.item())
        
        return action
        
    
    def replay(self, episode):
        if len(self.memory) < self.batch_size:
            return 
        
        self.mini_batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch(episode)
        
        self.expected_state_action_values = self.get_expected_state_action_values()
        
        self.update_q_network()
        
        
    def make_minibatch(self, episode):
    
        if episode < 5: 
            transitions = self.memory.sample(self.batch_size)
        else:
            indexes = self.td_error_memory.get_prioritized_indexes(self.batch_size)
            transitions = [self.memory.memory[n] for n in indexes]
            
            
        mini_batch = Transition(*zip(*transitions))
        #print("메모리에서 랜덤 샘플")
        
        state_batch = torch.cat(mini_batch.state)
        action_batch = torch.cat(mini_batch.action)
        reward_batch = torch.cat(mini_batch.reward)
        non_final_next_states = torch.cat([s for s in mini_batch.next_state if s is not None])
        
        return mini_batch, state_batch, action_batch, reward_batch, non_final_next_states
        
        
    def get_expected_state_action_values(self):
    
        self.model.eval()
        self.target_model.eval()
        
        #print(self.state_batch.shape)
        
        self.state_action_values = self.model(self.state_batch).gather(1, self.action_batch)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.mini_batch.next_state)), dtype=torch.bool).to(device)
        
        next_state_values = torch.zeros(self.batch_size).to(device)
        
        a_m = torch.zeros(self.batch_size,  dtype=torch.long).to(device)
        
        a_m[non_final_mask] = self.model(self.non_final_next_states).detach().max(1)[1]
        
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        
        next_state_values[non_final_mask] = self.target_model(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        
        expected_state_action_values = self.reward_batch + self.discount_factor*next_state_values
        
        return expected_state_action_values
        
        
    def update_q_network(self):
    
        self.model.train()
        self.loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        #loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1)) #원래 이거였음
        
        #print("모델 훈련")
        self.optimizer.zero_grad()
        self.loss.backward()
        #print("loss:%0.4f" % self.loss)
        #loss.backward() #원래 이거였음
        #for param in self.model.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target_q_network(self):
        #print("타겟모델 업데이트")
        self.target_model.load_state_dict(self.model.state_dict())       
            
        
    def update_td_error_memory(self): 
        '''TD 오차 메모리에 저장된 TD 오차를 업데이트'''
         
        self.model.eval()
        self.target_model.eval()
        
        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))
                
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.model(state_batch).gather(1, action_batch)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
        
        next_state_values = torch.zeros(len(self.memory)).to(device)
        a_m = torch.zeros(len(self.memory), dtype=torch.long).to(device)
        
        a_m[non_final_mask] = self.model(non_final_next_states).detach().max(1)[1]
        
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        
        next_state_values[non_final_mask] = self.target_model(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        td_errors = (reward_batch + self.discount_factor * next_state_values) - state_action_values.squeeze()
            
        #rospy.loginfo("TD_ERRORS:%0.3f"%td_errors)
        
        self.td_error_memory.memory = td_errors.detach().numpy().tolist()
        
        
        

class Agent():
    def __init__(self, state_size, action_size):
        self.brain = Brain(state_size, action_size)
        
    def update_q_function(self, episode):
        self.brain.replay(episode)
        
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
        
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)        
        
    def update_target_q_function(self):
        self.brain.update_target_q_network()
                
    def memorize_td_error(self, td_error):  
        '''TD 오차 메모리에 TD 오차를 저장'''
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):  
        '''TD 오차 메모리의 TD 오차를 업데이트'''
        self.brain.update_td_error_memory()
    
    
    
    

if __name__ == '__main__':
    rospy.init_node('mobile_robot_dqn_lidar_per')
    
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    get_action = Float32MultiArray()
    
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_loss_result = rospy.Publisher('loss_result', Float32MultiArray, queue_size=5)
    
    result = Float32MultiArray()
    loss_result = Float32MultiArray()
    

    state_size = 214
    action_size = 7

    env = Env(action_size)
    
    agent = Agent(state_size, action_size)
    
    scores, losses, episodes = [], [], []
    global_step = 0
    start_time = time.time()
    
        
    for episode in range(agent.brain.load_episode + 1, EPISODES):
        #print("Episode:",episode)
        time_out = False
        done = False
        
        state = env.reset()
        #old_action = 3
        
        # print("Episode:",episode, "state:",state)
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0).to(device)
        
        score = 0
        losses = 0.0
        t = 0
        
        while True:
        #for t in range(agent.brain.episode_step):
            t += 1
            action = agent.get_action(state, episode)
            print("step: ", t, "   episode: ", episode)

            observation_next, reward, done = env.step(action.item())
            #print("Reward: ", reward)
            reward = (torch.tensor([reward]).type(torch.FloatTensor)).to(device)
            state_next = observation_next
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
            state_next = torch.unsqueeze(state_next, 0).to(device)
            
            agent.memorize(state, action, state_next, reward)
            
            agent.memorize_td_error(0)
            
            agent.update_q_function(episode)
            
            state = state_next
            old_action = action.item()
            
            score += reward
            losses += agent.brain.loss            
            
            get_action.data = [action.int(), score, reward.int()]
            pub_get_action.publish(get_action)        
            
            
            if t >= agent.brain.episode_step:
                rospy.loginfo("Time out!!")
                time_out = True
                
            if done:
                #agent.update_target_q_function()   
                #rospy.loginfo("UPDATE TARGET NETWORK")
        
                state_next = None
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f' % (episode, score, len(agent.brain.memory), agent.brain.epsilon))
                #scores.append(score)
                #episodes.append(episode)
                state = env.reset()
                # print("Episode:",episode, "state:",state)
                state = torch.from_numpy(state).type(torch.FloatTensor)
                state = torch.unsqueeze(state, 0).to(device)
            
            
            if time_out: 
                
                state_next = None
            
                #agent.update_target_q_function()
                #rospy.loginfo("UPDATE TARGET NETWORK")
                
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f' % (episode, score, len(agent.brain.memory), agent.brain.epsilon))
                
                scores.append(score)
                #losses.append(agent.brain.loss)
                episodes.append(episode)
                
                result.data = [score, episode] 
                loss_result.data = [losses/agent.brain.episode_step, episode]
                pub_result.publish(result)
                pub_loss_result.publish(loss_result)
                
                #state = env.reset()
                ## print("Episode:",episode, "state:",state)
                #state = torch.from_numpy(state).type(torch.FloatTensor)
                #state = torch.unsqueeze(state, 0).to(device)    
                
                
                break
            
        agent.update_td_error_memory()
        
        agent.update_target_q_function()   
        rospy.loginfo("UPDATE TARGET NETWORK")
        
        if agent.brain.epsilon > agent.brain.epsilon_min:
            agent.brain.epsilon *= agent.brain.epsilon_decay
            
        if episode % 2 == 0:
            #agent.update_target_q_function()
            #rospy.loginfo("UPDATE TARGET NETWORK")
            with torch.no_grad():
                torch.save(agent.brain.model, agent.brain.dirPath + str(episode) + '.pt') 
                
    with torch.no_grad():        
        torch.save(agent.brain.model, agent.brain.dirPath + str(episode) + '.pt')        
    print("종료")           
          

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
