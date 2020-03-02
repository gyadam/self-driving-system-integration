#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5 # Maximum deceleration to keep up a nice drive-behaviour

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Subscribe to ROS-topics to get current-position and waypoint information
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Create ROS-topic-publisher to pass the final_waypoints to drive to other nodes.
        # queue_size=1 to ensure non old informations is used by subscribers.
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_lane = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.loop()

     # Setup rospy-utils to keep the publishing at a specific frequency (in ours case 50x per seconds (20ms))
    def loop(self):
        rate = rospy.Rate(50)
        # Keep running until the node gets shutdown-signal
        while not rospy.is_shutdown():
            # Avoid race-conditions when base_waypoints or current position was not intialized
            if self.pose and self.base_lane:
                # Get waypoints to drive
                self.publish_waypoints()
            # Use ros-sleep-function to reach the required frequency
            rate.sleep()
            
    # Use hyperplane and current-car-position to get the closest waypoint to drive to
    def get_closest_waypoint_idx(self):
        # Get current car position on x-y-axis
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # Search for closest waypoint in KD-tree: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.query.html
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Get the closest and previous coord to check if the closest coords in infront or behind us
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)  # closest waypoint
        prev_vector = np.array(prev_coord) # waypoint befoe closest
        pos_vect = np.array([x, y])        # our position

        # Use hyperplane to find out if the given waypoint in infront or behind us which indicates if 
        # we should consider the waypoint in our path-planning or use the next waypoint. Visualization:
        # https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Hesse_normalenform.svg/1200px-Hesse_normalenform.svg.png
        dot_product = np.dot(cl_vect-prev_vector, pos_vect-cl_vect)

        # If closest waypoint is behind vehicle use waypoint beyond the closest
        if dot_product > 0:
            # Use modulo operator to avoid out-of-range (restart at begin if end is reached)
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        
        return closest_idx

    # Send the waypoints to the subscribed notes
    def publish_waypoints(self, closest_idx):
        lane = Lane() # ROS Lane instance
        lane.header = self.base_waypoints.header # Reuse current header-information from basic waypoints
        # Add closest waypoint which is infront of us +LOOKAHEAD_WPS waypoint beyond 
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS] 
        self.final_waypoints_pub.publish(lane)

    # Refresh current car-position information
    def pose_cb(self, msg):
        self.pose = msg

    # Base-Waypoints are just published once. Ensure to store them in our variables
    def waypoints_cb(self, waypoints):
        # Store waypoints in local variable for class-instance
        self.base_waypoints = waypoints
        # Create the KDTree which help us to find the waypoint-informations way more efficient
        # KDTree well explained can be found here: https://www.youtube.com/watch?v=TLxWtXEbtFE
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
