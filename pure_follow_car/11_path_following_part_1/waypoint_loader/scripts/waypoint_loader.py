#!/usr/bin/env python

import os
import csv
import math

from geometry_msgs.msg import Quaternion, PoseStamped

from styx_msgs.msg import Lane, Waypoint

from nav_msgs.msg import Path

import tf
import rospy

CSV_HEADER = ['x', 'y', 'yaw']
MAX_DECEL = 1.0


class WaypointLoader(object):

    def __init__(self):
        rospy.init_node('waypoint_loader', log_level=rospy.DEBUG)
        # 第一个pub是真正要使用读取的消息类型(gazebo)，是自定义的msg(styx_msgs)
        # 而第二个因为rviz只支持标准Path的消息格式，仅仅为了在rviz中可视化路径 可以不用
        self.pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1, latch=True)
        self.pub_path = rospy.Publisher('/base_path', Path, queue_size=1, latch=True)
        # 从参数服务器获取要达到的额定速度(10km/h)和csv文件所描述的路径(点(x,y,yaw)) 其在waypoint.launch文件中指定
        self.velocity = self.kmph2mps(rospy.get_param('~velocity'))
        self.new_waypoint_loader(rospy.get_param('~path'))
        rospy.spin()

    def new_waypoint_loader(self, path):
        if os.path.isfile(path):
            waypoints, base_path = self.load_waypoints(path)
            self.publish(waypoints, base_path)
            rospy.loginfo('Waypoint Loded')
        else:
            rospy.logerr('%s is not a file', path)

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)
    # fname为路径(csv)文件名
    def load_waypoints(self, fname):
        waypoints = []
        base_path = Path()
        base_path.header.frame_id = 'world'
        # 读取csv文件的固定三行写法  with…as语句是简化版的try except finally语句
        """ csv.DictReader()返回一个csv.DictReader对象，可以将读取的信息映射为字典，其关键字由可选参数fieldnames来指定。
            csvfile：可以是文件(file)对象或者列表(list)对象。
            fieldnames：是一个序列，用于为输出的数据指定字典关键字，如果没有指定，则以第一行的各字段名作为字典关键字。
            dialect：编码风格，默认为excel的风格
        """
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            for wp in reader:
                # Waypoint是自定义的消息类型(类)pose是路径点的位置，twist是车在路径点应达到的速度(10km/h) forward是朝向
                p = Waypoint()
                p.pose.pose.position.x = float(wp['x'])
                p.pose.pose.position.y = float(wp['y'])
                # 存储pose时统一用四元数存储，因为欧拉角有万向锁问题
                q = self.quaternion_from_yaw(float(wp['yaw']))
                p.pose.pose.orientation = Quaternion(*q)
                p.twist.twist.linear.x = float(self.velocity)
                p.forward = True
                # 将csv文件每一行添加到list 
                waypoints.append(p)
                # 存储到base_path用于在rviz中可视化 
                # PoseStamped：header(seq stamp frame_id) pose(position(x,y,z) oreaintation(x,y,z,w))
                path_element = PoseStamped()
                path_element.pose.position.x = p.pose.pose.position.x
                path_element.pose.position.y = p.pose.pose.position.y
                base_path.poses.append(path_element)

   
        waypoints = self.decelerate(waypoints)
        return waypoints,base_path

    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)
    # 当小车快要到达终点时需要速度慢慢为零
    # 遍历waypoints的每个点求与最后一个点的距离，再转换为速度vel和liner.x(10km/h)进行比较取二者较小者重新赋给waypoints列表
    # 这样就进行了在该点处指定的速度慢慢递减至0，而不是一直为10km/h   ，注意读取后速度进行了转换(m/s)
    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints
    # 发布lane(gazebo路径)和(base_path)rviz可视化路径 ，/world同样是为了使rviz中的基准坐标系不是小车本身
    def publish(self, waypoints, base_path):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints  # 因为lane.msg定义的的wagpoints是Waypoint[] waypoints(list类型)
        self.pub.publish(lane)
        self.pub_path.publish(base_path)


if __name__ == '__main__':
    try:
        WaypointLoader()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint node.')
