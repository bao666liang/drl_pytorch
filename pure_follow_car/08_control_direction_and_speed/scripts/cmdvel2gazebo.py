#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import math

class CmdVel2Gazebo:

    def __init__(self):
        rospy.init_node('cmdvel2gazebo', anonymous=True)
        
        rospy.Subscriber('/smart/cmd_vel', Twist, self.callback, queue_size=1)

        self.pub_steerL = rospy.Publisher('/smart/front_left_steering_position_controller/command', Float64, queue_size=1)
        self.pub_steerR = rospy.Publisher('/smart/front_right_steering_position_controller/command', Float64, queue_size=1)
        self.pub_rearL = rospy.Publisher('/smart/rear_left_velocity_controller/command', Float64, queue_size=1)
        self.pub_rearR = rospy.Publisher('/smart/rear_right_velocity_controller/command', Float64, queue_size=1)

        # initial velocity and tire angle are 0
        self.x = 0
        self.z = 0

        # car Wheelbase (in m) 轴距L
        self.L = 1.868

        # car Tread 轮距T
        self.T_front = 1.284
        self.T_rear = 1.284 #1.386

        # how many seconds delay for the dead man's switch 一旦一段时间未收到信号则要停车 0.2s
        self.timeout=rospy.Duration.from_sec(0.2)
        self.lastMsg=rospy.Time.now()

        # maximum steer angle of the "inside" tire 车身和轮向最大角度 (str_angle = 0.6rad)
        self.maxsteerInside=0.6

        # turning radius for maximum steer angle just with the inside tire 单车模型最大alpha下转弯半径
        # tan(maxsteerInside) = wheelbase/radius --> solve for max radius at this angle
        rMax = self.L/math.tan(self.maxsteerInside)

        # radius of inside tire is rMax, so radius of the ideal middle tire (rIdeal) is rMax+treadwidth/2
        rIdeal = rMax+(self.T_front/2.0)  # 四轮中轴转弯半径

        # maximum steering angle for ideal middle tire # 阿克曼转向
        # tan(angle) = wheelbase/radius # 中轴最大接受+/-alpha(外部输入的) 
        self.maxsteer=math.atan2(self.L,rIdeal)

        # loop
        rate = rospy.Rate(10) # run at 10Hz
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()
        

    def callback(self,data):
        # w = v / r 因为后轮驱动，R为后轮半径，又速度发布单位为弧度，要转换为角速度发布(转速)
        self.x = data.linear.x / 0.3
        # constrain the ideal steering angle such that the ackermann steering is maxed out
        self.z = max(-self.maxsteer,min(self.maxsteer,data.angular.z))
        self.lastMsg = rospy.Time.now() # 记录何时收到这条msg，为了计算publish()长时间接受不到指令去停止小车 0.2s

    def publish(self):
        # now that these values are published, we
        # reset the velocity, so that if we don't hear new
        # ones for the next timestep that we time out; note
        # that the tire angle will not change
        # NOTE: we only set self.x to be 0 after 200ms of timeout
        delta_last_msg_time = rospy.Time.now() - self.lastMsg
        msgs_too_old = delta_last_msg_time > self.timeout
        if msgs_too_old: # 后轮至零，前轮归正，Float64()是pub定义的类
            self.x = 0
            msgRear = Float64()
            msgRear.data = self.x
            self.pub_rearL.publish(msgRear)
            self.pub_rearR.publish(msgRear)
            msgSteer = Float64()
            msgSteer.data = 0
            self.pub_steerL.publish(msgSteer)
            self.pub_steerR.publish(msgSteer)

            return

        # The self.z is the delta angle in radians of the imaginary front wheel of ackerman model.
        if self.z != 0:
            T_rear = self.T_rear
            T_front = self.T_front
            L=self.L
            # self.v is the linear *velocity*
            r = L/math.fabs(math.tan(self.z))
            # 四个轮子分别的转弯半径 r为不同z下的中轴转弯半径,z为中轴转弯角度
            rL_rear = r-(math.copysign(1,self.z)*(T_rear/2.0))
            rR_rear = r+(math.copysign(1,self.z)*(T_rear/2.0))
            rL_front = r-(math.copysign(1,self.z)*(T_front/2.0))
            rR_front = r+(math.copysign(1,self.z)*(T_front/2.0))
            msgRearR = Float64()
            # 差速器原理
            # the right tire will go a little faster when we turn left (positive angle)
            # amount is proportional to the radius of the outside/ideal 
            # 后两轮的转速 左转时右轮大于左轮转速
            msgRearR.data = self.x*rR_rear/r
            msgRearL = Float64()
            # the left tire will go a little slower when we turn left (positive angle)
            # amount is proportional to the radius of the inside/ideal
            msgRearL.data = self.x*rL_rear/r

            self.pub_rearL.publish(msgRearL)
            self.pub_rearR.publish(msgRearR)

            msgSteerL = Float64()
            msgSteerR = Float64()
            # the left tire's angle is solved directly from geometry
            # 左转时前两轮转角 左转时左轮转角要略大于右轮
            msgSteerL.data = math.atan2(L,rL_front)*math.copysign(1,self.z)
            self.pub_steerL.publish(msgSteerL)
    
            # the right tire's angle is solved directly from geometry
            msgSteerR.data = math.atan2(L,rR_front)*math.copysign(1,self.z)
            self.pub_steerR.publish(msgSteerR)
        else:
            # if we aren't turning
            # 如果 z = 0则车辆保持直行
            msgRear = Float64()
            msgRear.data = self.x
            self.pub_rearL.publish(msgRear)
            # msgRear.data = 0;
            self.pub_rearR.publish(msgRear)

            msgSteer = Float64()
            msgSteer.data = self.z

            self.pub_steerL.publish(msgSteer)
            self.pub_steerR.publish(msgSteer)


if __name__ == '__main__':
    try:
        CmdVel2Gazebo()
    except rospy.ROSInterruptException:
        pass


