#!/usr/bin/env python

import os
import numpy as np
from math import cos, sin

from geometry_msgs.msg import PoseStamped, TwistStamped

from gazebo_msgs.msg import ModelStates, LinkStates

import tf
import rospy 
# 除了gps slam 粒子滤波获取定位，gazebo仿真时会有msg实时发布小车位置和速度(/gazebo/model_states)：代码订阅gazebo信息，做数据转换，发布转换后信息smart
# 在现实中不会有这种实现，但做循线没必要做定位
# rostopic echo /smart/center_Pose   /smart/rear_pose 车辆中心和后轴中心(方便路径规划)
# 其数据格式:PoseStamped, TwistStamped为带时间戳(header)的速度（twist）和位置（四元数表示位置和朝向）信息
class vehicle_pose_and_velocity_updater:
	def __init__(self):
		rospy.init_node('vehicle_pose_and_velocity_updater', log_level=rospy.DEBUG)

		self.rear_pose_pub = rospy.Publisher('/smart/rear_pose', PoseStamped, queue_size = 1)
		self.center_pose_pub = rospy.Publisher('/smart/center_pose', PoseStamped, queue_size = 1)
		self.vel_pub = rospy.Publisher('/smart/velocity', TwistStamped, queue_size = 1)
		# 每收到一帧消息做一次处理
		rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_cb, queue_size = 1)

		rospy.spin()

	def model_cb(self,data):
		# ModelStates(data)有name pose twist 如果smart未spwan则return
		try:
			vehicle_model_index = data.name.index("smart")
		except:
			return
		vehicle_position = data.pose[vehicle_model_index]
		vehicle_velocity = data.twist[vehicle_model_index]
		orientation = vehicle_position.orientation
		(_, _, yaw) = tf.transformations.euler_from_quaternion([orientation.x,orientation.y,orientation.z,orientation.w])
		time_stamp = rospy.Time.now()

		# vehicle center position
		center_pose = PoseStamped()
		center_pose.header.frame_id = '/world' # frame_id 即下面定义的位置和朝向都是关于world坐标系发布的 用于rviz
		center_pose.header.stamp = time_stamp  # 时间戳，获取当前时间
		center_pose.pose.position = vehicle_position.position
		center_pose.pose.orientation = vehicle_position.orientation
		self.center_pose_pub.publish(center_pose)

		# vehicle rear axle position
		rear_pose = PoseStamped()
		rear_pose.header.frame_id = '/world'
		rear_pose.header.stamp = time_stamp
		center_x = vehicle_position.position.x
		center_y = vehicle_position.position.y
		rear_x = center_x - cos(yaw) * 0.945
		rear_y = center_y - sin(yaw) * 0.945
		rear_pose.pose.position.x = rear_x
		rear_pose.pose.position.y = rear_y
		rear_pose.pose.orientation = vehicle_position.orientation
		self.rear_pose_pub.publish(rear_pose)

		# vehicle velocity
		velocity = TwistStamped()
		velocity.header.frame_id = ''
		velocity.header.stamp = time_stamp
		velocity.twist.linear = vehicle_velocity.linear
		velocity.twist.angular = vehicle_velocity.angular
		self.vel_pub.publish(velocity)



if __name__ == "__main__":
	try:
		vehicle_pose_and_velocity_updater()
	except:
		rospy.logwarn("cannot start vehicle pose and velocity updater updater")