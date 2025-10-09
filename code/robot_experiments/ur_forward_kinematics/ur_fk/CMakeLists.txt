cmake_minimum_required(VERSION 3.8)
project(ur_fk)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Add the message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointPoses.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Include directories for the generated headers
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Export rosidl runtime dependencies
ament_export_dependencies(rosidl_default_runtime)

# Add the executable
add_executable(ur5e src/ur5e.cpp)
ament_target_dependencies(ur5e rclcpp Eigen3 geometry_msgs sensor_msgs std_msgs)

# Link the message dependencies to the executable
rosidl_target_interfaces(ur5e ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install the executable
install(TARGETS
  ur5e
  DESTINATION lib/${PROJECT_NAME}/
)

# Install the source directory
install(
  DIRECTORY
    src
  DESTINATION
    share/${PROJECT_NAME}/
)

ament_package()
