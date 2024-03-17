# Task 1 - Multi-Robot Simulation 

## Part (A): Optimize the URDF and Simulator

For this part, I simply removed the camera (all related joints, links, and plugins) from the URDF file of the Voltas robot to reduce the computational effort on Gazebo. I also removed some unnecessary visual tags for easy and smooth simulation.

Also added the Differential drive plugin to the URDF to get access to the motion for the robots, and changed the URDF to load the lidar, imu, and diff_drive plugin for all robots with unique remapped topics with their namespaces.

## Part (B): Setting up the multi-robot simulation

This was a pretty challenging task, so this is the solution I came up with - created three separate launch files:

1. Launch file to spawn a single robot with input pose and name_space. Also included the joint_state_publisher for each of the robots using name_space.

2. Launch file to spawn multiple robots using grouping with unique name_spaces and tf_prefixes with input pose for all of them. This launch file spawns 5 Voltas robots into Gazebo. (This launch is modular and can include n number of bots in the future).

3. A main launch file that opens an empty Gazebo world and calls the multi_bot.launch file.

The next task is to set up navigation stacks for all the robots, using the same concepts of remapping the required topics to their respective namespaces. Created launch files for navigation to control all the robots with each having their nav stack (amcl_launch, move_base_launch files for all the bots and a main nav_launch file).

## Part (C): Writing the Controller

So I was able to successfully load the robots into the gazebo with their sensor, cmd_vel, and Odom topics. But when I would publish commands on the topics the bot wouldn't move, I tried a lot but couldn't do anything. For the time being, I wrote a simple Artificial Potential Field algorithm for moving bots from high potential to low potential, using the attraction and repulsion model.

# Task 2 - 3D Object Detection:

## Part (A): Converting dataset to .bag files

Downloaded a raw sample KITTI dataset of around (0.6) GB, used kitti2bag to convert them into .bag file. Also prepared a launch file to visualize the dataset using the camera and the PointCloud2 messages on Rviz and also used rosbag play -l to play the bag file in a loop to publish the data.

## Part (B): Implement PointNet on ROS

Firstly after a lot of time researching found a PointNet Frustum Pre-trained model trained on sample KITTI raw datasets. Loaded the model using the PyTorch library into the code.

Secondly wrote a Python script initializing a ROS node subscribing to the topic "/kitti/velo/pointcloud" on which the bag file advertises its data. The next step was to pre-process the data, the raw data format having the shape as just the number of points but the required input was (batch, 4, number of points). So performed basic numpy padding and dimension extensions. Also, the input points for each point cloud were 1024 so altered the code in such a way that it loops through each point cloud with increments of 1024 and inputs it to the model. After various iterations of data pre-processing the data was fit to run the model, The model ran and gave the following outputs:

1. Translation Vectors - (x,y,z) co-ordinates w.r.t to the local frame
2. Bounding Box Parameters - Box parameters (Couldn't post-process)
3. Segmented Point Clouds Mean

As the model was too complicated, I couldn’t post-process the model outputs to Make the bounding boxes and visualize lidar-based 3-D object detection. I tried finding the center of the detected objects and marking it in Rviz using PointStamped but it wasn't accurate enough.
