cmake_minimum_required(VERSION 3.8)
project(low_level_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(kinenik REQUIRED)
find_package(custom_interfaces REQUIRED)
find_package(urdf REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_msgs)
find_package(sensor_msgs)
find_package(control_msgs)
#find_package(robot_arm_interfaces)

install(
  DIRECTORY
    config
    launch
    src
    urdf
    worlds
  DESTINATION
    share/${PROJECT_NAME}/
)

add_executable(cartesian_position_controller src/cartesian_position_controller.cpp)
target_include_directories(cartesian_position_controller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(cartesian_position_controller rclcpp rclcpp_components rclcpp_action trajectory_msgs geometry_msgs kinenik custom_interfaces control_msgs)

#add_executable(madar_trajectory_controller src/madar_trajectory_controller.cpp)
#target_include_directories(madar_trajectory_controller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
#ament_target_dependencies(madar_trajectory_controller rclcpp rclcpp_components rclcpp_action trajectory_msgs geometry_msgs robot_arm_interfaces custom_interfaces control_msgs)


install(TARGETS
  cartesian_position_controller
  #madar_trajectory_controller
  DESTINATION lib/${PROJECT_NAME})

ament_package()
