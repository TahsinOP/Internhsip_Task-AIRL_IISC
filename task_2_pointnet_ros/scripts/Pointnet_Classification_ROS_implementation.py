#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import tensorflow as tf
import numpy as np
from huggingface_hub import from_pretrained_keras


class PointNetModel:
    def __init__(self):
        # self.model_path = '/home/tahsin/catkin_ws/src/task_2_pointnet_ros'
        self.model = from_pretrained_keras("keras-io/PointNet")
        print(self.model.summary())

    def predict(self, data):
        # Assuming data is preprocessed and in the correct format
        return self.model.predict(data)

class PointNetNode:
    def __init__(self):
        self.model = PointNetModel()
        self.subscriber = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.point_cloud_callback)

    
    def pad_last_window(self, window_points, window_size):
        if len(window_points) < window_size:
            padding = np.zeros((window_size - len(window_points), 3))
            window_points = np.concatenate((window_points, padding), axis=0)
        return window_points



    def point_cloud_callback(self, msg):
        try:
            points = np.array(list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            print(f"Total points: {len(points)}")

            # Define the window size
            window_size = 2048

            # Initialize an empty list to store predictions
            predictions = []

            # Process the point cloud in windows of 2048 points
            for i in range(0, len(points), window_size):
                window_points = points[i:i+window_size]
                window_points = self.pad_last_window(window_points, window_size)

                # Add a batch dimension
                window_points = np.expand_dims(window_points, axis=0)
                print(np.shape(window_points))
                window_points = window_points / np.linalg.norm(window_points, axis=1, keepdims=True)

                # Make a prediction for the current window
                window_prediction = self.model.predict(window_points)
                predictions.append(window_prediction)

            # Process and publish the results or print them (not implemented)
            print(predictions)
            
        except Exception as e:
            print(f"Error processing point cloud: {e}")

def main():
    rospy.init_node('pointnet_node')
    node = PointNetNode() # Specify the path to your model here
    rospy.spin()

if __name__ == '__main__':
    main()
