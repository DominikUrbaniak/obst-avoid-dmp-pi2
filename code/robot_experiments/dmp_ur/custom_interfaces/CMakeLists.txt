cmake_minimum_required(VERSION 3.8)
project(custom_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ReachPoint.srv"
  "srv/ExecuteTrajectory.srv"
  "srv/ExecuteRrtTrajectory.srv"
  "srv/BoxWallConfig.srv"
  "srv/WallsGapConfig.srv"
  "srv/InsertConfig.srv"
  "srv/ExecuteDmp.srv"
  "srv/ExecuteDmpAuto.srv"
  "srv/GetAvoidanceParams.srv"
  "srv/GetBoundingBoxParams.srv"
  "srv/CustomService.srv"
  "srv/CheckTrajectory.srv"
  "msg/CustomMessage.msg"

  DEPENDENCIES geometry_msgs sensor_msgs trajectory_msgs

 )

ament_package()
