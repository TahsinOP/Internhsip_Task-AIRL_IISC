#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import tensorflow as tf
import numpy as np

class PointNetModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        # Assuming data is preprocessed and in the correct format
        return self.model.predict(data)

class PointNetNode:
    def __init__(self, model_path):
        self.model = PointNetModel(model_path)
        self.subscriber = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, msg):
        points = np.array(list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        # normalize the points
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        predictions = self.model.predict(points)
        # Process and publish the results or print them
        print(predictions)

def main():
    rospy.init_node('pointnet_node')
    node = PointNetNode('/src') # Specify the path to your model here
    rospy.spin()

if __name__ == '__main__':
    main()
