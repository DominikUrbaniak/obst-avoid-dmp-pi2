cmake_minimum_required(VERSION 3.8)
project(ur5e_moveit_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_common REQUIRED)
find_package(custom_interfaces REQUIRED)

add_executable(ur5e_trajectory_node src/ur5e_trajectory_node.cpp)
ament_target_dependencies(ur5e_trajectory_node rclcpp moveit_ros_planning_interface moveit_common moveit_msgs custom_interfaces)


install(TARGETS
  ur5e_trajectory_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
