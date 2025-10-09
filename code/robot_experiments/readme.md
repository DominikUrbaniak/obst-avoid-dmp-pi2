## Pick-and-Drop experiment
Implementation of the pick-and-drop experiment with a UR5e manipulator, Robotiq 2F-85 gripper, using Ubuntu 22.04, ROS 2 Humble and Gazebo 11.

## External ROS 2 Packages
(to be included into the workspace along the packages from this repository)
- ROS2 UR description package (ur_description), humble branch (-b humble): https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- IOC-UPC inverse kinematics library for UR robots (kinenik): https://gitioc.upc.edu/robots/kinenik
- Gazebo ROS Link Attacher for simulating stable grasp (gazebo_ros_link_attacher), humble-devel branch (-b humble-devel): https://github.com/davidorchansky/gazebo_ros_link_attacher

## Other package requirements
- DMP library: https://github.com/mginesi/dmp_pp
- PyTorch for mapping datasets to neural networks
- Open3D for processing point clouds

## Packages in this repository
- low_level_control: Setup of the gazebo environment with robot, gripper and objects, and JointTrajectory controller, using modified launch file from https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation
- ur5e_moveit_commander: Alternative low-level control via MoveIt
- high_level_control: Provides services to start the experiment, generates the DMP trajectories from the neural network models, and provides gripper controller
- image_processing: Subscribes and processes point cloud
- custom_interfaces: Includes the custom messages and services
- robotiq_2f_model: Model from https://github.com/beta-robots/robotiq/tree/master/robotiq_2f_model wrapped in ROS2 package
- ur_moveit_config: MoveIt configuration created with the Setup Assistant https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
- ur_forward_kinematics: publishes all robot joints enabling be removal of the robot manipulator from the point cloud
- (docs: target folder for logging experiments)

## Run experiment
Terminal 1
- ros2 launch low_level_control main_avoid_wall_3p.launch.py

Terminal 2

Option 1:
- copy dmp_ur/docs/bash/run_pickndrop.sh and experiment_list.json in workspace directory
- chmod +x run_pickndrop.sh
- ./run_pickndrop.sh <experiment_index>

Option 2
- ros2 service call /high_level_control/go_to_start std_srvs/srv/Empty
- ros2 param set /high_level_controller drop_height_offset 0.07
- ros2 param set /move_group use_sim_time True
- ros2 param set /obstacle_detection visualize_obst_view_all True
- ros2 service call /high_level_control/set_obstacle_config_cup custom_interfaces/srv/BoxWallConfig "{start_pose: {position: {x: -0.4, y: 0.5}}, first_x: 0.4, cup_x: 0.3, cup_y: 0.5, ny: 4, nz: 2}"
- ros2 service call /low_level_control/reach_point custom_interfaces/srv/ReachPoint "{x: 0.4, y: -0.5, z: 0.18, qr: 0.0, qp: 3.141592653589793, qy: 1.5707963268}"
- ros2 service call /high_level_control/execute_pick_and_drop custom_interfaces/srv/ExecuteDmpAuto "{obj_dims: [0.07, 0.07, 0.13], sol_id: -1}"
