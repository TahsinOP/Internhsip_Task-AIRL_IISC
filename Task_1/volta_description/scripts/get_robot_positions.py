#!/usr/bin/env python3
# generate_launch.py

launch_template = """
<launch>
    <arg name="model" default="$(find volta_description)/urdf/volta.xacro"/>
    <arg name="viz_config" default="$(find volta_description)/rviz_params/urdf.rviz"/>
    <param name="robot_description" command="$(find xacro)/xacro --inorder $(arg model)"/>
    <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
    <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" />
    <include file="$(find gazebo_ros)/launch/empty_world.launch">
        <arg name="world_name" value="$(find gazebo_ros)/worlds/empty.world"/>
    </include>
"""

robot_template = """
    <group ns="robot{index}">
        <param name="tf_prefix" value="robot{index}_tf" />
        <include file="$(find volta_description)/launch/single_bot.launch" >
            <arg name="index" value="{index}" />
            <arg name="init_pose" value="-x {x} -y {y} -z {z}" />
            <arg name="robot_name" value="Robot{index}" />
        </include>
    </group>
"""

launch_end = """
</launch>
"""

def generate_launch_file(num_robots):
    launch_content = launch_template
    for i in range(1, num_robots + 1):
        x, y, z = (i - 1, 0, 0)  # Adjust the initial poses as needed
        robot_block = robot_template.format(index=i, x=x, y=y, z=z)
        launch_content += robot_block

    launch_content += launch_end
    with open("multi_robot_launch.launch", "w") as launch_file:
        launch_file.write(launch_content)

if __name__ == "__main__":
    num_robots = 5  # Change this to the desired number of robots
    generate_launch_file(num_robots)

