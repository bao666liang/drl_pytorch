#!/usr/bin/env python

import os
import csv
import math

from geometry_msgs.msg import Quaternion, PoseStamped, TwistStamped, Twist

from styx_msgs.msg import Lane, Waypoint

from gazebo_msgs.msg import ModelStates

import tf
import rospy

"""vscode注释：
单行注释：ctrl+/ 多行注释：alt+shit+A
取消注释: ctrl+/ 多行注释：alt+shit+A
"""
"""如果出现不能加载控制器的报错，类似于【ERROR】 【1586241514.504639, 148.457000】: Failed to load joint_state_controller，
请依照ros版本用sudo apt install的方法安装以下三个包（以melodic版本为例）：
1. sudo apt install ros-noetic-gazebo-ros-control
2. sudo apt install ros-noetic-ros-control
3. sudo apt install ros-noetic-ros-controllers
"""

# 从自行车模型出发，纯追踪算法以车后轴为切点、车辆纵向车身为切线，
# 通过控制前轮转向角，使车辆可以沿着一条经过目标路经点的圆弧行驶
# 前视距离(是路径长度，不一定是直线，可以根据速度或曲率实时变化，这里设为定值)
HORIZON = 6.0

class PurePersuit:
	def __init__(self):
		rospy.init_node('pure_persuit', log_level=rospy.DEBUG)
		# 获取小车后轮中轴位置和局部路径点（21个点->20m）
		rospy.Subscriber('/smart/rear_pose', PoseStamped, self.pose_cb, queue_size = 1)
		# rospy.Subscriber('/smart/velocity', TwistStamped, self.vel_cb, queue_size = 1)
		rospy.Subscriber('/final_waypoints', Lane, self.lane_cb, queue_size = 1)
		# 控制小车的速度和前轮(方向盘)转角theta  /smart/cmd_vel是前几节讲的 差速器原理
		self.twist_pub = rospy.Publisher('/smart/cmd_vel', Twist, queue_size = 1)

		self.currentPose = None
		self.currentVelocity = None
		self.currentWaypoints = None

		self.loop()

	def loop(self):
		rate = rospy.Rate(20)
		rospy.logwarn("pure persuit starts")
		while not rospy.is_shutdown():
			if self.currentPose and self.currentVelocity and self.currentWaypoints:
				twistCommand = self.calculateTwistCommand()
				self.twist_pub.publish(twistCommand)
			rate.sleep()

	def pose_cb(self,data):
		self.currentPose = data

	# def vel_cb(self,data):
	# 	self.currentVelocity = data

	def lane_cb(self,data):
		self.currentWaypoints = data

	def calculateTwistCommand(self):
		lad = 0.0 #look ahead distance accumulator
		# 前视距离小于6米时就一直将局部路径的最后一个点作为后轮前视目标点，所以一开始targetIndex=20
		targetIndex = len(self.currentWaypoints.waypoints) - 1
		# 在局部路径里从起始点开始做一个距离累加计算出最终lad即为前视距离(局部路径会跟着小车走因此根据当前小车位置累加一次即可)
		for i in range(len(self.currentWaypoints.waypoints)): # range(21) 遍历0-20不包括21(21个点)
			if((i+1) < len(self.currentWaypoints.waypoints)):
				this_x = self.currentWaypoints.waypoints[i].pose.pose.position.x
				this_y = self.currentWaypoints.waypoints[i].pose.pose.position.y
				next_x = self.currentWaypoints.waypoints[i+1].pose.pose.position.x
				next_y = self.currentWaypoints.waypoints[i+1].pose.pose.position.y
				lad = lad + math.hypot(next_x - this_x, next_y - this_y)
				if(lad > HORIZON):
					targetIndex = i+1
					break


		targetWaypoint = self.currentWaypoints.waypoints[targetIndex]
		# 将局部路径的第一个点的速度作为要发布的速度
		targetSpeed = self.currentWaypoints.waypoints[0].twist.twist.linear.x
		# 获取当前小车的位置和目标点的位置
		targetX = targetWaypoint.pose.pose.position.x
		targetY = targetWaypoint.pose.pose.position.y		
		currentX = self.currentPose.pose.position.x
		currentY = self.currentPose.pose.position.y
		#get vehicle yaw angle
		quanternion = (self.currentPose.pose.orientation.x, self.currentPose.pose.orientation.y, self.currentPose.pose.orientation.z, self.currentPose.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quanternion)
		yaw = euler[2] # (0.,0.,yaw)
		#get angle difference 计算弦切角alpha，要减去汽车基于world的朝向角
		alpha = math.atan2(targetY - currentY, targetX - currentX) - yaw
		l = math.sqrt(math.pow(currentX - targetX, 2) + math.pow(currentY - targetY, 2))
		# 因为小车走到最后时即使将所有点加起来也不会大于前向距离，这时发布指令是速度和前轮转角为0
		# 大于的话将纯追踪算法公式计算出前轮的偏转角和参考速度发布 ，l是目标点和车后轮中轴点的距离
		if(l > 0.5):
			theta = math.atan(2 * 1.868 * math.sin(alpha) / l)
			# #get twist command
			twistCmd = Twist()
			twistCmd.linear.x = targetSpeed
			twistCmd.angular.z = theta 
		else:
			twistCmd = Twist()
			twistCmd.linear.x = 0
			twistCmd.angular.z = 0

		return twistCmd


if __name__ == '__main__':
    try:
        PurePersuit()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start motion control node.')

