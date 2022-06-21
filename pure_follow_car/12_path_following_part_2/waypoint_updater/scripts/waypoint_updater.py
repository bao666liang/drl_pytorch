#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from nav_msgs.msg import Path

import math
import numpy as np
from scipy.spatial import KDTree

# 小车向前看的距离20m(20个坐标点 因为每个坐标点距离1m) (实际上车速越快或曲率越大看的距离越远，但这里简化取定值)
LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        # 订阅小车当前位置(后轮中轴点)  全局路径
        rospy.Subscriber('/smart/rear_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # 发布局部路径让小车进行路径跟踪 Path数据类同样仅用于rviz可视化  话题名称可自定义，sub和pub相同即可
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1) # 局部路径点发布(真正用的)
        self.final_path_pub = rospy.Publisher('final_path', Path, queue_size=1) # 填充好的局部路径发布(rviz可视化用的)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.pose = None

        # rospy.spin() 0.05s
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # 所以用None初始化为了返回bool 若接收到小车位置，全局路径，KD树生成则生成final_path并发布
            if self.pose and self.base_waypoints and self.waypoint_tree:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
    # callback:接收到小车位置存入pose变量
    def pose_cb(self, msg):
        self.pose = msg
    # callback：接收全局路径点列表并将其转换为KDtree数据结构类型
    # KDtree:其结构方便快速查找与小车位置最近的点进而计算距离，虽然其生成比较慢但默认只生成一次
    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
    # 用KDtree的query()获取与小车最近的点的索引
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        return closest_idx          
    # 从全局路径点列表中实时计算出与小车最近的点，向后选取20个点即为局部路径点(Lane()存储)
    def publish_waypoints(self, closest_idx):
    	# fill in final waypoints to publish
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]

        # 将局部路径点转化为路径(Path()存储)在rviz可视化填充
        path = Path()
        path.header.frame_id = '/world'
        for p in lane.waypoints:
            path_element = PoseStamped()
            path_element.pose.position.x = p.pose.pose.position.x
            path_element.pose.position.y = p.pose.pose.position.y
            path.poses.append(path_element)

        self.final_waypoints_pub.publish(lane)
        self.final_path_pub.publish(path)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
