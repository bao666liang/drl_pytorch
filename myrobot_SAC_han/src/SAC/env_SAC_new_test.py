#!/usr/bin/env python3

import rospy
import numpy as np
import math
import os
import time
import random
from math import pi, sqrt, pow, exp
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Point, Pose  # 速度(v,w)  点  位置(x,y,zeta)消息对象
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# std_srvs 包含两种服务类型，称为 Empty 和 Trigger
# 对于 Empty 服务，服务和客户端之间不交换任何实际数据 , Trigger 服务增加了检查触发是否成功的可能性 
    
def goal_def():

    obstacles = [[2,0],[4,0],[3,3],[0,2],[0,4],[-2,0],[-4,0],[-3,3],[0,-2],[0,-4],[3,-3],[-3,-3]]
    while True:
        distance = []
        # print("finding proper goal position...")
        # 计算随机产生的目标点与每个障碍物的距离 距离大于一且不是0  然后返回目标点坐标
        x, y = np.random.randint(-40,40)/10, np.random.randint(-40,40)/10
        for i in range(len(obstacles)):
            distance.append(sqrt(pow(obstacles[i][0]-x,2)+pow(obstacles[i][1]-y,2)))
        if (min(distance) >= 1.0) and (not(x==0 and y==0)):
            break
              
    return x, y 


class Env():
    def __init__(self):
    
        #self.RespawnGoal = respawn_goal()
        self.delete = False
        self.goal_x, self.goal_y = 4.5, 4.5 # goal_def()
        #self.goal_x, self.goal_y = self.RespawnGoal.goal_def(self.delete)
        
        self.heading = 0
        #self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.line_error = 0
        
        self.current_obstacle_angle = 0
        self.old_obstacle_angle = 0
        self.current_obstacle_min_range = 0
        self.old_obstacle_min_range = 0
        self.t = 0
        self.old_t = time.time()
        self.dt = 0
        
        self.time_step = 0  # 时间步
        # 服务客户端：重置仿真过程(包括时间) 恢复物理更新 暂停物理更新(仿真时间没有改变) 
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        
        # 퍼블리셔    参数(话题，消息类型) -> 实例化 cmd_vel = Twist()即消息对象用于获取数据
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        # 서브스크라이버
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.getScanData) 
        self.scan = LaserScan()
    # 将订阅到的一维雷达数据赋给变量self.scan
    def getScanData(self, scan_msg):
        self.scan = scan_msg        

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 4)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        # 小车转到目标点所需的角度 (-pi,pi)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        #self.heading = round(heading, 2)
        self.heading = heading

    def getState(self, scan, ep):
        # scan.ranges 距离数组，长度360
        scan_range = []        
        heading = self.heading
        min_range = 0.20 # 障碍物大小(圆)
        done = False
        arrival = False
        #target_size = 1.0        
        # ep是第几个episode targer_size是目标点大小(圆形)
        if ep<30: target_size = 1.2
        elif (ep>=30 and ep<100): target_size = 0.7
        else: target_size = 0.5
        # samples = 210 这里将12个障碍物当作质点处理
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(12) # 地图为10*10 将12m添加进对应位置以防算不出来
            elif np.isnan(scan.ranges[i]): # np.isnan()是判断是否是空值 NaN 然后用0代替
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
                
        # 四舍五入保留小数点后两位并取其索引   
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        # 碰撞(圆形 r=0.2)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        
        #print("obstacle_min_range", obstacle_min_range)
        #print("obstacle_angle", obstacle_angle)
        #print("current: %0.2f, %0.2f" %(self.position.x, self.position.y))
        
        #print("current_distance:",current_distance)
        # 到达目标点
        if current_distance < target_size:
            self.get_goalbox = True
            arrival = True
            # done = True
        # 跑出地图   
        boundary = math.sqrt(pow(self.position.x,2) + pow(self.position.y,2))
        if boundary > 10:
            #done = False
            done = True
            
        #print("laser scan:",scan_range)

        #return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done
        # 朝目标要转过的角度，当前与目标点的直线距离，和最近障碍点的距离，和最近障碍点的角度(数组索引表示)
        state = [heading, current_distance, obstacle_min_range, obstacle_angle]
        #print("state", state)
        return state, done, arrival 

    # def setReward(self, state, done, action): ## 2022 . 02 . 13 예전거 comment out 
    
    #     yaw_reward = []
        
    #     obstacle_angle = state[3]
    #     obstacle_min_range = state[2]
    #     current_distance = state[1]
    #     heading = state[0]      
        
        
    #     #angular_rate = abs(action[0])/1.5       
    #     #linear_rate  = action[1]/0.5
        
    #     #angular_reward = -2*angular_rate + 2
    #     #linear_reward  =  2*linear_rate
    #     #action_reward  = angular_reward + linear_reward
        
    #     if (obstacle_min_range < 0.6):
    #         obstacle_reward = -20/(obstacle_min_range+0.0001)
    #     else:
    #         obstacle_reward = 0.0
            
    #     lin_vel_rate = action[1]/0.5
    #     ang_vel_rate = abs(action[0])/1.5 
    #     lin_vel_reward = 3*lin_vel_rate
    #     ang_vel_reward = -2*ang_vel_rate+2
    #     action_reward = ang_vel_reward + lin_vel_reward
        
    #     distance_rate = (current_distance / (self.goal_distance+0.0000001))
    #     distance_reward = -10*distance_rate+10 if distance_rate<=1 else 1
        
    #     angle_rate = 2*abs(heading)/pi
    #     angle_reward = -10*angle_rate+10 if angle_rate<=1 else -5*angle_rate+5
        
    #     #time_reward = -self.time_step/20
    #     #time_reward = 1/(self.time_step +1)
        
    #     reward = distance_reward * angle_reward + obstacle_reward
        
    #     if done:
    #         #rospy.loginfo("Collision!")
    #         reward = -5000
    #         self.time_step = 0
    #         self.pub_cmd_vel.publish(Twist())

    #     if self.get_goalbox:
    #         rospy.loginfo("Goal!!")
    #         reward = 4000
    #         self.pub_cmd_vel.publish(Twist())
    #         self.delete = True
    #         #self.goal_x, self.goal_y = self.RespawnGoal.goal_def(self.delete)
    #         self.goal_x, self.goal_y = goal_def() #4.5, 4.5 # goal_def()
    #         print("NEXT GOAL : ", self.goal_x, self.goal_y )
    #         self.goal_distance = self.getGoalDistace()
    #         self.get_goalbox = False
    #         self.time_step = 0
    #         #time.sleep(0.2)
            
    #     #print("total Reward:%0.3f"%reward, "\n")
    #     #print("Reward : ", reward)
        
    #     return reward


    def setReward(self, state, done, action):
    
        yaw_reward = []
        
        obstacle_angle = state[3]
        obstacle_min_range = state[2]
        current_distance = state[1]
        heading = state[0]      
        
        
        #angular_rate = abs(action[0])/1.5       
        #linear_rate  = action[1]/0.5
        
        #angular_reward = -2*angular_rate + 2
        #linear_reward  =  2*linear_rate
        #action_reward  = angular_reward + linear_reward
        # 底盘：0.45 0.3 0.21 障碍物：0.2 预留：0.7-0.2-0.3=0.2
        if (obstacle_min_range < 0.7):
            obstacle_reward = -0.5/(obstacle_min_range+0.0001)
        else:
            obstacle_reward = 0.0
        # action=[角速度，线速度]   
        lin_vel_rate = action[1]/0.5
        ang_vel_rate = abs(action[0])/1.5 
        lin_vel_reward = 3*lin_vel_rate
        ang_vel_reward = -2*ang_vel_rate+2
        action_reward = ang_vel_reward + lin_vel_reward
        
        distance_rate = (current_distance / (self.goal_distance+0.0000001))
        distance_reward = -2*distance_rate+2 if distance_rate<=1 else 1
        
        angle_rate = 2*abs(heading)/pi
        angle_reward = -3*angle_rate+3 if angle_rate<=1 else -1*angle_rate+1
        
        #time_reward = -self.time_step/20
        #time_reward = 1/(self.time_step +1)
        time_reward = -2
        
        reward = distance_reward * angle_reward + obstacle_reward + time_reward
        
        if done: # 碰撞后奖励-100，step归零，发布速度信息
            #rospy.loginfo("Collision!")
            reward = -100
            self.time_step = 0
            self.pub_cmd_vel.publish(Twist())
        # get_goalbox一开始为false，值为true则到达目标点,然后再值为false
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.delete = True
            #self.goal_x, self.goal_y = self.RespawnGoal.goal_def(self.delete)
            self.goal_x, self.goal_y = goal_def() #4.5, 4.5 # goal_def()
            print("NEXT GOAL : ", self.goal_x, self.goal_y )
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            self.time_step = 0
            #time.sleep(0.2)
            
        #print("total Reward:%0.3f"%reward, "\n")
        #print("Reward : ", reward)
        
        return reward

    def step(self, action, ep):
        
        self.time_step += 1
        
        vel_cmd = Twist()
        vel_cmd.linear.x  =  action[1] #0.5
        vel_cmd.angular.z = action[0]  # action
        self.pub_cmd_vel.publish(vel_cmd)
        
        #print("EP:", ep, " Step:", t, " Goal_x:",self.goal_x, "  Goal_y:",self.goal_y)
        
        state, done, arrival = self.getState(self.scan, ep)
        reward = self.setReward(state, done, action)
        # 将结构数据转化为ndarray  主要区别在于 np.array（默认情况下）将会copy该对象，而 np.asarray除非必要，否则不会copy该对象
        return np.asarray(state), reward, done, arrival 

    def reset(self, ep):
    
        self.time_step = 0
        # 等待重置话题服务启动成功
        rospy.wait_for_service('gazebo/reset_simulation')
        #time.sleep(0.1)
        try: 
            self.reset_proxy()  # 创建客户端请求对象:重置仿真环境服务
            #time.sleep(0.1)
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
            
        while True:
            if (len(self.scan.ranges) > 0): # 等待雷达返回距离信息
                break
        #error_data = self.line_error
        self.goal_distance = self.getGoalDistace()
        state, done, _ = self.getState(self.scan, ep)
             
        

        return np.asarray(state)
        
        
        
        
     
