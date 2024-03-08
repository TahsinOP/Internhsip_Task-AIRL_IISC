import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from point_model import PointNetModel
import numpy as np

# Pre - Process the point cloud data coming from the dataset .bag file 

def point_cloud_callback(msg):

    points = np.array(list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    # normalize the points
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    predictions = PointNetModel.predict(points)


def main():

    rospy.init_node('pointnet_node')
    rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, point_cloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
