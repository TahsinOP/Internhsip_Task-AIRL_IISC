#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import torch
import torch.nn as nn
import numpy as np
from frustum_pointnet import FrustumPointNet

# Adjust the path to where your Frustum-PointNet model is saved


class FrustumPointNetModel:
    def __init__(self):
        # Assuming you have a PyTorch model class defined somewhere
        # For example, if your model class is named FrustumPointNetModelPyTorch
        model_path = '/home/tahsin/3DOD_thesis/pretrained_models/model_37_2_epoch_400.pth'
        model_id = "frustum_pointnet_v1" # Example model ID
        project_dir = "home/tahsin/3DOD_thesis" # Path to your project directory
        num_points = 1024

        self.model = FrustumPointNet(model_id,project_dir,num_points)
        self.model = self.model.cpu()
        self.NH = self.model.BboxNet_network.NH
        self.regression_loss_func = nn.SmoothL1Loss()
        self.model.load_state_dict(torch.load((model_path),map_location=torch.device('cpu')))
        
        self.model.eval()
         # Set the model to evaluation mode
        print(self.model)

    def predict(self, data):
        # Convert data to PyTorch tensor and move to the correct device
        data_tensor = torch.from_numpy(data).float()

        # print(data_tensor)

        with torch.no_grad():
            out_InstanceSeg, out_TNet, out_BboxNet, out_seg_point_clouds_mean, out_dont_care_mask = self.model(data_tensor)

            out_InstanceSeg = out_InstanceSeg.cpu().numpy()
            out_TNet = out_TNet.cpu().numpy()
            out_BboxNet = out_BboxNet.cpu().numpy()
            out_seg_point_clouds_mean = out_seg_point_clouds_mean.cpu().numpy()
            out_dont_care_mask = out_dont_care_mask.cpu().numpy()

            print("Instance Segmentation Masks:")
            for mask in out_InstanceSeg[0]:
                print(f" {mask}")

            # Print translation vectors
            print("Translation Vectors:")
            for vec in out_TNet:
                print(f" {vec}")

            # Print bounding box parameters
            print("Bounding Box Parameters:")
            for bbox in out_BboxNet[0]:
                print(f" {bbox}")

            # Print segmented point clouds mean
            print("Segmented Point Clouds Mean:")
            for mean in out_seg_point_clouds_mean[0]:
                print(f" {mean}")

            # Print don't care mask
            print("Don't Care Mask:")
            print(f" {out_dont_care_mask[0]}")
            # print(f"The predictions are {prediction}")
        

class FrustumPointNetNode:
    def __init__(self):
        self.model = FrustumPointNetModel()
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
            # print(np.shape(points))

            # Define the window size
            window_size = 1024

            # Initialize an empty list to store predictions
            predictions = []

            # Process the point cloud in windows of 1024 points
            for i in range(0, len(points), window_size):
                window_points = points[i:i+window_size]
                window_points = self.pad_last_window(window_points, window_size)

                # Convert to homogeneous coordinates
                point_cloud_xyz_hom = np.ones((window_points.shape[0], 4))
                point_cloud_xyz_hom[:, 0:3] = window_points

                # Normalize the points
                # point_cloud_xyz_hom[:, 0:3] = point_cloud_xyz_hom[:, 0:3] / np.linalg.norm(point_cloud_xyz_hom[:, 0:3], axis=1, keepdims=True)

                # Transpose the data to match the expected input shape (batch_size, 4, num_points)
                point_cloud_xyz_hom = point_cloud_xyz_hom.transpose(1, 0)

                # Add a batch dimension
                point_cloud_xyz_hom = np.expand_dims(point_cloud_xyz_hom, axis=0)

                print(np.shape(point_cloud_xyz_hom))

                # Make a prediction for the current window
                window_prediction = self.model.predict(point_cloud_xyz_hom)
                
                print(f"the predictions are {window_prediction}")
                # predictions.append(window_prediction)

            # Process and publish the results or print them (not implemented)
            # print(f"the predications are {predictions}")
        
        except Exception as e:
            print(f"Error processing point cloud: {e}")
    
        


def main():
    rospy.init_node('frustum_pointnet_node')

    node = FrustumPointNetNode()
    rospy.spin()

if __name__ == '__main__':
    main()
