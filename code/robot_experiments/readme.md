## Pick-and-Drop experiment
Implementation of the pick-and-drop experiment with a UR5e manipulator, Robotiq 2F-85 gripper, using Ubuntu 22.04, ROS 2 Humble and Gazebo 11.

## External ROS 2 Packages
(to be included into the workspace along the packages from this repository)
- ROS2 UR description package (ur_description), humble branch (-b humble): https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- IOC-UPC inverse kinematics library for UR robots (kinenik): https://gitioc.upc.edu/robots/kinenik
- Gazebo ROS Link Attacher for simulating stable grasp (gazebo_ros_link_attacher), humble-devel branch (-b humble-devel): https://github.com/davidorchansky/gazebo_ros_link_attacher

## Packages in this repository
- low_level_control: Setup of the gazebo environment with robot, gripper and objects, and FollowJointTrajectory , using modified launch file from https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation
- high_level_control: Provides services to start the experiment, generates the DMP trajectories from the neural network models
- image_processing: Subscribes and processes point cloud
- custom_interfaces: Includes the custom messages and services
- robotiq_2f_model: Model from https://github.com/beta-robots/robotiq/tree/master/robotiq_2f_model wrapped in ROS2 package
- ur_moveit_config: MoveIt configuration created with the Setup Assistant https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
- docs: target folder for logging experiments
