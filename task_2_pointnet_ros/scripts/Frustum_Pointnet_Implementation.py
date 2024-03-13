#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import torch
import torch.nn as nn
import numpy as np
from frustum_pointnet import FrustumPointNet


class FrustumPointNetModel:
    def __init__(self):

        model_path = '/home/tahsin/3DOD_thesis/pretrained_models/model_37_2_epoch_400.pth'
        model_id = "frustum_pointnet_v1" # Example model ID
        project_dir = "home/tahsin/3DOD_thesis" # Path to your project directory
        num_points = 2048

        self.model = FrustumPointNet(model_id,project_dir,num_points)
        self.model = self.model.cpu()
        self.NH = self.model.BboxNet_network.NH
        self.regression_loss_func = nn.SmoothL1Loss()
        self.model.load_state_dict(torch.load((model_path),map_location=torch.device('cpu')))
        
        self.model.eval()
        print(self.model)

    def predict(self, data):

        data_tensor = torch.from_numpy(data).float()
        with torch.no_grad():
            out_InstanceSeg, out_TNet, out_BboxNet, out_seg_point_clouds_mean, out_dont_care_mask = self.model(data_tensor)
            out_InstanceSeg = out_InstanceSeg.cpu().numpy()
            out_TNet = out_TNet.cpu().numpy()
            out_BboxNet = out_BboxNet.cpu().numpy()
            out_seg_point_clouds_mean = out_seg_point_clouds_mean.cpu().numpy()
            out_dont_care_mask = out_dont_care_mask.cpu().numpy()

            # Print translation vectors
            print("Translation Vectors:")
            for vec in out_TNet:
                print(f" {vec}")
                self.object_marker = vec

            # Print bounding box parameters
            print("Bounding Box Parameters:")
            for bbox in out_BboxNet[0]:
                print(f" {bbox}")
                self.Bbox_out = out_BboxNet[0]

            # Print segmented point clouds mean
            print("Segmented Point Clouds Mean:")
            for mean in out_seg_point_clouds_mean[0]:
                print(f" {mean}")
            
        return vec ,out_dont_care_mask   

class FrustumPointNetNode:
    def __init__(self):
        self.model = FrustumPointNetModel()
        self.subscriber = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.point_cloud_callback)
        self.marker_pub = rospy.Publisher('/object_center_markers', PointStamped, queue_size=10)

    def pad_last_window(self, window_points, window_size):
        if len(window_points) < window_size:
            padding = np.zeros((window_size - len(window_points), 3))
            window_points = np.concatenate((window_points, padding), axis=0)
        return window_points

    def point_cloud_callback(self, msg):
        try:
            points = np.array(list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
            print(f"Total points: {len(points)}")
            window_size = 1024
            
            # Process the point cloud in windows of 1024 points
            for i in range(0, len(points), window_size):
                window_points = points[i:i+window_size]
                self.x = window_points[i//1024][0]
                self.y = window_points[i//1024][1]
                self.z = window_points[i//1024][2]
                window_points = self.pad_last_window(window_points, window_size)

                # Convert to homogeneous coordinates
                point_cloud_xyz_hom = np.ones((window_points.shape[0], 4))
                point_cloud_xyz_hom[:, 0:3] = window_points
                print(point_cloud_xyz_hom)
                point_cloud_xyz_hom = point_cloud_xyz_hom.transpose(1, 0)

                # Add a batch dimension
                point_cloud_xyz_hom = np.expand_dims(point_cloud_xyz_hom, axis=0)

                # Make a prediction for the current window
                window_prediction , no_object  = self.model.predict(point_cloud_xyz_hom)
               
                if no_object == 1 : 

                    object_center = self.model.object_marker
                    Bbox_in = np.array(self.model.Bbox_out)
                    print(Bbox_in)
                    self.publish_marker(object_center=object_center ,Bbox_in= Bbox_in)

                print(f"the predictions are {window_prediction}")
        
        except Exception as e:
            print(f"Error processing point cloud: {e}")

    def publish_marker(self, object_center,Bbox_in):
        # Create a PointStamped message to represent the object center
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "base_link"  # Adjust frame_id if necessary

        # Set the point coordinates
        point_msg.point.x = float(-Bbox_in[0]-object_center[0]+self.x)
        point_msg.point.y = float( -Bbox_in[1]-object_center[1]+self.y)
        point_msg.point.z = float( -Bbox_in[2]-object_center[2]+self.z)

        # Publish the marker message
        self.marker_pub.publish(point_msg)
    
def main():
    rospy.init_node('frustum_pointnet_node')
    node = FrustumPointNetNode()
    rospy.spin()

if __name__ == '__main__':
    main()
