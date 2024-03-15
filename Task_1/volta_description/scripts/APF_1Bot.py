#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

class APF:
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.goal_position = PoseStamped() # Set your goal position here
        self.goal_position.pose.position.x = 10.0 # Example goal position
        self.goal_position.pose.position.y = 10.0
        self.goal_position.pose.position.z = 0.0

        self.robot_positions = [] # List to store positions of all robots

    def odom_callback(self, msg):
        # Assuming each robot has a unique odom topic
        # You might need to adjust this based on your setup
        self.robot_positions.append(msg.pose.pose.position)

        # Calculate forces based on current position, goal position, and obstacles
        # This is a simplified example. You'll need to implement the full APF logic here.
        force_x = self.goal_position.pose.position.x - msg.pose.pose.position.x
        force_y = self.goal_position.pose.position.y - msg.pose.pose.position.y

        # Normalize forces
        force_magnitude = (force_x**2 + force_y**2)**0.5
        force_x /= force_magnitude
        force_y /= force_magnitude

        # Publish Twist messages to cmd_vel topic
        twist = Twist()
        twist.linear.x = force_x
        twist.linear.y = force_y
        self.cmd_vel_pub.publish(twist)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('apf_node')
    apf = APF()
    apf.run()