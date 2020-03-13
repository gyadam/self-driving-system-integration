#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # Vehicle HYPERPARAMETERS
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        ## Publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        ## Variables
        # Input
        self.current_vel = None
        self.linear_vel = None
        self.angular_vel = None
        self.dbw_enabled = None

        # Output
        self.throttle = 0
        self.steering = 0
        self.brake = 0

        # Controller object
        self.controller = Controller(wheel_base, steer_ratio, max_lat_accel, max_steer_angle, decel_limit, vehicle_mass, wheel_radius)

        ## Subscribers
        rospy.Subscriber("/current_velocity", TwistStamped, self.callback_current_vel)
        rospy.Subscriber("/twist_cmd", TwistStamped, self.callback_twist)
        rospy.Subscriber("/vehicle/dbw_enabled", Bool, self.callback_dwb_enabled)

        self.loop()

    def loop(self):
        '''
        Loop for drive by wire node
        '''
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # Run controller, if all data is avaiable
            if not None in (self.current_vel, self.linear_vel, self.angular_vel):
                self.throttle, self.brake, self.steering = self.controller.control(self.current_vel, self.linear_vel, self.angular_vel, self.dbw_enabled)
            
            # Only publish, if drive-by-wire is enabled
            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)

            rate.sleep()

    def callback_current_vel(self, msg):
        '''
        Callback function for current velocity subscriber; extracts information from TwistStamped message
        '''
        self.current_vel = msg.twist.linear.x

    def callback_twist(self, msg):
        '''
        Callback function for twist subscriber; extracts information from TwistStamped message
        '''
        self.linear_vel = msg.twist.linear.x
        self.angular_vel = msg.twist.angular.z

    def callback_dwb_enabled(self, msg):
        '''
        Callback function for dbw enabled subscriber; extracts information from TwistStamped message
        '''
        self.dbw_enabled = msg

    def publish(self, throttle, brake, steer):
        '''
        Function to convert values from controller to ROS messages
        '''
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
