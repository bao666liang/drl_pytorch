#!/usr/bin/env python  
import rospy
import tf
from geometry_msgs.msg import PoseStamped
# rviz通过tf信息实现数据可视化 (robot_state_publish)  rosrun rqt_tf_tree rqt_tf_tree
# rviz模型加载通过robot_description
# 记得将control.launch的注释取消
class transform_publisher():
	def __init__(self):
		rospy.init_node('transform_publisher')
		# 上一讲所定义好的车辆的位置和朝向信息
		rospy.Subscriber('/smart/center_pose', PoseStamped, self.pose_cb, queue_size = 1)

		rospy.spin()
		# frame_id:父坐标 /world  child_id:子坐标 /base_link  定义两者转换关系
		# 使小车在rviz中Fixed Frame 的base_link变为world，不会静止而是绕着world固定坐标系运动
		# 因为tf中只有小车自身部件的tf关系，要加上世界坐标(虚拟坐标系)小车才会(看起来)运动
		# world名称可以自定义，但要同另一个.py中的frame_id相同
	def pose_cb(self, msg):
		pose = msg.pose.position
		orientation = msg.pose.orientation
		br = tf.TransformBroadcaster()
		br.sendTransform((pose.x, pose.y, pose.z),
							(orientation.x, orientation.y, orientation.z, orientation.w),
							rospy.Time.now(),
							'base_link', 'world')


if __name__ == "__main__":
	try:
		transform_publisher()
	except:
		rospy.logwarn("cannot start transform publisher")