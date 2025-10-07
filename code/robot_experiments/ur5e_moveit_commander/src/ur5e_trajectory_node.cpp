#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include "custom_interfaces/srv/execute_rrt_trajectory.hpp"
#include "custom_interfaces/srv/check_trajectory.hpp"
#include <std_msgs/msg/bool.hpp>

// Define joint limits for UR5e (radians/s and radians/s²)
std::map<std::string, double> max_velocities = {
  {"shoulder_pan_joint", 3.15},
  {"shoulder_lift_joint", 3.15},
  {"elbow_joint", 3.15},
  {"wrist_1_joint", 3.2},
  {"wrist_2_joint", 3.2},
  {"wrist_3_joint", 3.2}
};

std::map<std::string, double> max_accelerations = {
  {"shoulder_pan_joint", 9.0},
  {"shoulder_lift_joint", 9.0},
  {"elbow_joint", 9.0},
  {"wrist_1_joint", 10.0},
  {"wrist_2_joint", 10.0},
  {"wrist_3_joint", 10.0}
};

const int max_vel_factor = 1.0;
const int max_attempts = 20;
const int max_planning_time = 1.0;
const bool ignore_collisions = false;
const double obj_z_from_eef = 0.17; //0.18 is colliding with the floor (0.173 also), 0.17 works

class TrajectoryPlannerNode : public rclcpp::Node
{
public:

  TrajectoryPlannerNode(const rclcpp::NodeOptions& options) : Node("trajectory_planner_node", options),
    move_group_(std::shared_ptr<rclcpp::Node>(std::move(this)), "ur_arm"),
    planning_scene_interface_()
  {
    using std::placeholders::_1;
    using std::placeholders::_2;

    service_ = this->create_service<custom_interfaces::srv::ExecuteRrtTrajectory>(
      "/ur5e_moveit_commander/execute_rrt_trajectory",
      std::bind(&TrajectoryPlannerNode::handle_request, this, _1, _2));

    service_check_ = this->create_service<custom_interfaces::srv::CheckTrajectory>(
        "/ur5e_moveit_commander/check_trajectory",
        std::bind(&TrajectoryPlannerNode::handle_trajectory_request, this, _1, _2));

    trajectory_complete_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/low_level_control/trajectory_complete", 10);

  }

private:
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Service<custom_interfaces::srv::ExecuteRrtTrajectory>::SharedPtr service_;
  rclcpp::Service<custom_interfaces::srv::CheckTrajectory>::SharedPtr service_check_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trajectory_complete_pub_;

  void handle_request(
    const std::shared_ptr<custom_interfaces::srv::ExecuteRrtTrajectory::Request> request,
    std::shared_ptr<custom_interfaces::srv::ExecuteRrtTrajectory::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Entering moveit planning service...");
    move_group_.clearPathConstraints();

    // Remove all previously added obstacles (clean scene)
    auto existing = planning_scene_interface_.getKnownObjectNames();
    if (!existing.empty()) {
      planning_scene_interface_.removeCollisionObjects(existing);
      RCLCPP_INFO(this->get_logger(), "Cleared %zu previous obstacles from the planning scene.", existing.size());
    }

    // 1. Add bounding boxes as collision objects
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    for (size_t i = 0; i < request->obstacle_poses.size(); ++i)
    {
      moveit_msgs::msg::CollisionObject obj;
      obj.id = "obstacle_" + std::to_string(i);
      obj.header.frame_id = move_group_.getPlanningFrame();

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = primitive.BOX;
      primitive.dimensions = {
        request->obstacle_sizes[i].x,
        request->obstacle_sizes[i].y,
        request->obstacle_sizes[i].z
      };

      obj.primitives.push_back(primitive);
      obj.primitive_poses.push_back(request->obstacle_poses[i]);
      obj.operation = obj.ADD;

      collision_objects.push_back(obj);
    }

    // --- Define workspace bounds ---
    const double x_min = -0.9, x_max = 0.9;
    const double y_min = -0.2, y_max = 1.5;
    const double z_min =  -0.005, z_max = 0.8;
    const double thickness = 0.01; // wall thickness

    auto add_wall = [&](const std::string& id, double x, double y, double z,
                        double dx, double dy, double dz)
    {
      moveit_msgs::msg::CollisionObject wall;
      wall.id = id;
      wall.header.frame_id = move_group_.getPlanningFrame();

      shape_msgs::msg::SolidPrimitive shape;
      shape.type = shape.BOX;
      shape.dimensions = {dx, dy, dz};

      geometry_msgs::msg::Pose pose;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = z;
      pose.orientation.w = 1.0;

      wall.primitives.push_back(shape);
      wall.primitive_poses.push_back(pose);
      wall.operation = wall.ADD;

      collision_objects.push_back(wall);
    };

    // +X wall
    add_wall("wall_x_max", x_max + thickness / 2.0, (y_min + y_max) / 2, (z_min + z_max) / 2,
             thickness, y_max - y_min, z_max - z_min);
    // -X wall
    add_wall("wall_x_min", x_min - thickness / 2.0, (y_min + y_max) / 2, (z_min + z_max) / 2,
             thickness, y_max - y_min, z_max - z_min);
    // +Y wall
    add_wall("wall_y_max", (x_min + x_max) / 2, y_max + thickness / 2.0, (z_min + z_max) / 2,
             x_max - x_min, thickness, z_max - z_min);
    // -Y wall
    add_wall("wall_y_min", (x_min + x_max) / 2, y_min - thickness / 2.0, (z_min + z_max) / 2,
             x_max - x_min, thickness, z_max - z_min);
    // +Z ceiling
    add_wall("wall_z_max", (x_min + x_max) / 2, (y_min + y_max) / 2, z_max + thickness / 2.0,
             x_max - x_min, y_max - y_min, thickness);
    // Optional -Z floor
    add_wall("wall_z_min", (x_min + x_max) / 2, (y_min + y_max) / 2, z_min - thickness / 2.0,
             x_max - x_min, y_max - y_min, thickness);

    //planning_scene_interface_.applyCollisionObjects(collision_objects);
    //moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface_.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(this->get_logger(), "Added %zu obstacles to the planning scene", collision_objects.size());

    // Define the grasped object
    moveit_msgs::msg::CollisionObject grasped_object;
    grasped_object.id = "grasped_item";
    grasped_object.header.frame_id = move_group_.getEndEffectorLink();

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {0.02, 0.02, 0.02}; // example dimensions (width, depth, height)

    geometry_msgs::msg::Pose object_pose;
    object_pose.orientation.w = 1.0;
    object_pose.position.x = 0.0;
    object_pose.position.y = 0.0;
    object_pose.position.z = obj_z_from_eef; // half the height to center it

    grasped_object.primitives.push_back(primitive);
    grasped_object.primitive_poses.push_back(object_pose);
    grasped_object.operation = grasped_object.ADD;

    // Add it to the planning scene
    planning_scene_interface_.applyCollisionObjects({grasped_object});

    // Attach it to the robot
    move_group_.attachObject(
      grasped_object.id,
      move_group_.getEndEffectorLink(),
      {"gripper_left_pad", "gripper_right_pad"}
    );

    //geometry_msgs::msg::Pose current_pose = move_group_.getCurrentPose().pose;
    geometry_msgs::msg::Pose target_pose = request->target_pose;

    /*moveit_msgs::msg::OrientationConstraint ocm;
    ocm.link_name = move_group_.getEndEffectorLink();
    ocm.header.frame_id = move_group_.getPlanningFrame();
    ocm.orientation = target_pose.orientation;
    ocm.absolute_x_axis_tolerance = 0.2;
    ocm.absolute_y_axis_tolerance = 0.2;
    ocm.absolute_z_axis_tolerance = 0.2;
    ocm.weight = 1.0;

    moveit_msgs::msg::Constraints path_constraints;
    path_constraints.orientation_constraints.push_back(ocm);
    move_group_.setPathConstraints(path_constraints);*/

    // 2. Set pose target

    //target_pose.orientation = current_pose.orientation;  // optional
    move_group_.setPoseTarget(target_pose);
    //move_group_.setPoseTarget(request->target_pose);
    move_group_.setMaxVelocityScalingFactor(max_vel_factor);  // 50% of max velocity
    move_group_.setPlanningTime(max_planning_time);
    //move_group_.setWorkspace(-0.7, -0.0, 0.0, 0.7, 1.0, 1.0);

    //move_group_.setPlanningPipelineId("ompl");
    //move_group_.setPlannerId("RRTStar");



    bool success = false;
    int attempts = 0;
    auto start = this->now();
    rclcpp::Time end;
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    while (rclcpp::ok() && !success && attempts < max_attempts)
    {

      success = (move_group_.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
      end = this->now();

      //move_group_.clearPathConstraints();
      attempts++;
      if (!success) {
        RCLCPP_WARN(this->get_logger(), "Attempt %d failed, retrying...", attempts);
      }
    }

    // 4. Fill response
    response->success = success;
    response->planning_time = (end - start).seconds();
    response->attempts = attempts;
    response->planner = "rrt";

    if (success)
    {
      std_msgs::msg::Bool plan_msg;
      plan_msg.data = false;
      trajectory_complete_pub_->publish(plan_msg);
      RCLCPP_INFO(this->get_logger(), "Published false to /low_level_control/trajectory_complete from the ur5e_moveit_commander to show that the plan is successful and ready to be executed...");
      const trajectory_msgs::msg::JointTrajectory& traj = plan.trajectory_.joint_trajectory;
      response->rrt_joint_traj = traj;

      move_group_.execute(plan); //asyncExecute

      std_msgs::msg::Bool done_msg;
      done_msg.data = true;
      trajectory_complete_pub_->publish(done_msg);
      RCLCPP_INFO(this->get_logger(), "Published /low_level_control/trajectory_complete from the ur5e_moveit_commander");

      RCLCPP_INFO(this->get_logger(), "Motion planning succeeded in %.2f seconds", response->planning_time);
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Motion planning failed.");
    }

    /*
    const auto& q_curr = current_pose.orientation;
    RCLCPP_INFO(this->get_logger(), "Current EE orientation (quaternion): [x=%.3f, y=%.3f, z=%.3f, w=%.3f]",
            q_curr.x, q_curr.y, q_curr.z, q_curr.w);

    const auto& q_target = request->target_pose.orientation;
    RCLCPP_INFO(this->get_logger(), "Target EE orientation (quaternion): [x=%.3f, y=%.3f, z=%.3f, w=%.3f]",q_target.x, q_target.y, q_target.z, q_target.w);

    //rclcpp::Rate rate(10);
    int tries = 0;
    while (!move_group_.getCurrentState() && tries < 10) {
      RCLCPP_WARN(this->get_logger(), "Waiting for current robot state...");
      //rclcpp::spin_some(this);
      //rate.sleep();
      tries++;
    }
    if (!move_group_.getCurrentState()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to get current robot state after waiting.");
    } else {
      geometry_msgs::msg::Pose current_pose = move_group_.getCurrentPose().pose;
      const auto& q_curr = current_pose.orientation;
      RCLCPP_INFO(this->get_logger(),
                  "Current EE orientation (quaternion): [x=%.3f, y=%.3f, z=%.3f, w=%.3f]",
                  q_curr.x, q_curr.y, q_curr.z, q_curr.w);
    }*/


  }

  void handle_trajectory_request(
  const std::shared_ptr<custom_interfaces::srv::CheckTrajectory::Request> request,
  std::shared_ptr<custom_interfaces::srv::CheckTrajectory::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Received trajectory check request.");
    move_group_.clearPathConstraints();
    move_group_.setMaxVelocityScalingFactor(max_vel_factor);

    // Remove all previously added obstacles (clean scene)
    auto existing = planning_scene_interface_.getKnownObjectNames();
    if (!existing.empty()) {
      planning_scene_interface_.removeCollisionObjects(existing);
      RCLCPP_INFO(this->get_logger(), "Cleared %zu previous obstacles from the planning scene.", existing.size());
    }

    // 1. Add collision objects
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    for (size_t i = 0; i < request->obstacle_poses.size(); ++i) {
      moveit_msgs::msg::CollisionObject obj;
      obj.id = "obstacle_" + std::to_string(i);
      obj.header.frame_id = move_group_.getPlanningFrame();

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = primitive.BOX;
      primitive.dimensions = {
        request->obstacle_sizes[i].x,
        request->obstacle_sizes[i].y,
        request->obstacle_sizes[i].z
      };

      obj.primitives.push_back(primitive);
      obj.primitive_poses.push_back(request->obstacle_poses[i]);
      obj.operation = obj.ADD;
      collision_objects.push_back(obj);
    }

    // Define the grasped object
    moveit_msgs::msg::CollisionObject grasped_object;
    grasped_object.id = "grasped_item";
    grasped_object.header.frame_id = move_group_.getEndEffectorLink();

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {0.02, 0.02, 0.02}; // example dimensions (width, depth, height)

    geometry_msgs::msg::Pose object_pose;
    object_pose.orientation.w = 1.0;
    object_pose.position.x = 0.0;
    object_pose.position.y = 0.0;
    object_pose.position.z = obj_z_from_eef; // half the height to center it

    grasped_object.primitives.push_back(primitive);
    grasped_object.primitive_poses.push_back(object_pose);
    grasped_object.operation = grasped_object.ADD;

    // --- Define workspace bounds ---
    const double x_min = -0.9, x_max = 0.9;
    const double y_min = -0.3, y_max = 1.5;
    const double z_min =  -0.005, z_max = 0.8;
    const double thickness = 0.01; // wall thickness

    auto add_wall = [&](const std::string& id, double x, double y, double z,
                        double dx, double dy, double dz)
    {
      moveit_msgs::msg::CollisionObject wall;
      wall.id = id;
      wall.header.frame_id = move_group_.getPlanningFrame();

      shape_msgs::msg::SolidPrimitive shape;
      shape.type = shape.BOX;
      shape.dimensions = {dx, dy, dz};

      geometry_msgs::msg::Pose pose;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = z;
      pose.orientation.w = 1.0;

      wall.primitives.push_back(shape);
      wall.primitive_poses.push_back(pose);
      wall.operation = wall.ADD;

      collision_objects.push_back(wall);
    };

    // +X wall
    add_wall("wall_x_max", x_max + thickness / 2.0, (y_min + y_max) / 2, (z_min + z_max) / 2,
             thickness, y_max - y_min, z_max - z_min);
    // -X wall
    add_wall("wall_x_min", x_min - thickness / 2.0, (y_min + y_max) / 2, (z_min + z_max) / 2,
             thickness, y_max - y_min, z_max - z_min);
    // +Y wall
    add_wall("wall_y_max", (x_min + x_max) / 2, y_max + thickness / 2.0, (z_min + z_max) / 2,
             x_max - x_min, thickness, z_max - z_min);
    // -Y wall
    add_wall("wall_y_min", (x_min + x_max) / 2, y_min - thickness / 2.0, (z_min + z_max) / 2,
             x_max - x_min, thickness, z_max - z_min);
    // +Z ceiling
    add_wall("wall_z_max", (x_min + x_max) / 2, (y_min + y_max) / 2, z_max + thickness / 2.0,
             x_max - x_min, y_max - y_min, thickness);
    // Optional -Z floor
    add_wall("wall_z_min", (x_min + x_max) / 2, (y_min + y_max) / 2, z_min - thickness / 2.0,
             x_max - x_min, y_max - y_min, thickness);

    //planning_scene_interface_.applyCollisionObjects(collision_objects);

    // Add it to the planning scene
    planning_scene_interface_.applyCollisionObjects({grasped_object});

    // Attach it to the robot
    move_group_.attachObject(
      grasped_object.id,
      move_group_.getEndEffectorLink(),
      {"gripper_left_pad", "gripper_right_pad"}
    );

    //planning_scene_interface_.applyCollisionObjects(collision_objects);
    //moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface_.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(this->get_logger(), "Added %zu obstacles.", collision_objects.size());
    // 2. Attempt to compute Cartesian path
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;      // No jump tolerance
    const double eef_step = 0.01;           // Fine resolution
    std::vector<geometry_msgs::msg::Pose> waypoints = request->trajectory;

    auto start_time = this->now();
    double fraction = move_group_.computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory);
    auto end_time = this->now();

    // 3. Check if full path was valid
    response->planning_time = (end_time - start_time).seconds();
    response->attempts = 1; // We only try computeCartesianPath once here

    if ((fraction >= 0.99) || (ignore_collisions)) {
      if ((ignore_collisions) && (fraction < 0.99)){
        RCLCPP_WARN(this->get_logger(), "**Attention** Trajectory is executed even though collision were detected!: Fraction = %.2f", fraction);
      } else {
        RCLCPP_INFO(this->get_logger(), "Trajectory is collision-free");
      }// 3. Check velocity/acceleration limits
      auto start_time2 = this->now();
      if (check_trajectory_limits(trajectory.joint_trajectory, max_velocities, max_accelerations, this->get_logger())) {
        auto end_time2 = this->now();
        RCLCPP_INFO(this->get_logger(), "Trajectory is within velocity/acceleration limits.");
        RCLCPP_INFO(this->get_logger(), "Collision checks succeeded in %.2f seconds", response->planning_time);
        RCLCPP_INFO(this->get_logger(), "vel and acc checks succeeded in %.2f seconds", (end_time2 - start_time2).seconds());
        // 4. Execute
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        plan.trajectory_ = trajectory;
        response->rrt_joint_traj = trajectory.joint_trajectory;
        response->success = true;
        move_group_.execute(plan);

        std_msgs::msg::Bool done_msg;
        done_msg.data = true;
        trajectory_complete_pub_->publish(done_msg);
        return;
      } else {
        RCLCPP_WARN(this->get_logger(), "Velocity or acceleration limits violated in Cartesian path.");
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Unable to compute full Cartesian path. Fraction = %.2f", fraction);
      response->planner = "rrt";
      std_msgs::msg::Bool done_msg;
      done_msg.data = true;
      trajectory_complete_pub_->publish(done_msg);
      return;
    }


    // 4. Fall back to planning to the final pose
    move_group_.setPoseTarget(waypoints.back());
    move_group_.setPlanningTime(max_planning_time);


    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = false;
    int attempts = 0;
    //const int max_attempts = 20;

    while (rclcpp::ok() && !success && attempts < max_attempts) {
      success = (move_group_.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
      end_time = this->now();
      attempts++;

      if (!success)
        RCLCPP_WARN(this->get_logger(), "Planning attempt %d failed.", attempts);
    }

    response->success = success;
    response->attempts = attempts;
    response->planning_time = (end_time - start_time).seconds();
    response->planner = "rrt";

    if (success) {
      response->rrt_joint_traj = plan.trajectory_.joint_trajectory;
      move_group_.execute(plan);

      std_msgs::msg::Bool done_msg;
      done_msg.data = true;
      trajectory_complete_pub_->publish(done_msg);
      RCLCPP_INFO(this->get_logger(), "Planned and executed new trajectory.");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to plan a collision-free trajectory.");
    }

  }

  // Helper function to check velocity and acceleration
  bool check_trajectory_limits(const trajectory_msgs::msg::JointTrajectory& traj,
                               const std::map<std::string, double>& max_velocities,
                               const std::map<std::string, double>& max_accelerations,
                               rclcpp::Logger logger)
  {
    if (traj.points.size() < 2) {
      RCLCPP_WARN(logger, "Trajectory has fewer than 2 points, skipping limit checks.");
      return true;
    }

    for (size_t i = 1; i < traj.points.size(); ++i) {
      const auto& prev = traj.points[i - 1];
      const auto& curr = traj.points[i];
      rclcpp::Duration curr_time = rclcpp::Duration(curr.time_from_start);
      rclcpp::Duration prev_time = rclcpp::Duration(prev.time_from_start);
      double dt = (curr_time - prev_time).seconds();


      if (dt <= 0.0) {
        RCLCPP_ERROR(logger, "Non-positive time difference between trajectory points.");
        return false;
      }

      for (size_t j = 0; j < traj.joint_names.size(); ++j) {
        const std::string& joint_name = traj.joint_names[j];
        double dq = curr.positions[j] - prev.positions[j];
        double v = dq / dt;

        // Check velocity
        if (std::abs(v) > max_velocities.at(joint_name)) {
          RCLCPP_ERROR(logger, "Velocity limit exceeded for joint %s: %.3f > %.3f",
                       joint_name.c_str(), std::abs(v), max_velocities.at(joint_name));
          return false;
        }

        // Check acceleration if previous velocities exist
        if (!prev.velocities.empty() && !curr.velocities.empty()) {
          double dv = curr.velocities[j] - prev.velocities[j];
          double a = dv / dt;
          if (std::abs(a) > max_accelerations.at(joint_name)) {
            RCLCPP_ERROR(logger, "Acceleration limit exceeded for joint %s: %.3f > %.3f",
                         joint_name.c_str(), std::abs(a), max_accelerations.at(joint_name));
            return false;
          }
        }
      }
    }

    return true;
  }


};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);  // Initialize the ROS 2 system
  rclcpp::NodeOptions options;
  options.append_parameter_override("use_sim_time", true);

  auto node = std::make_shared<TrajectoryPlannerNode>(options);

  // Create the executor and add the node to it
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  // Spin the executor to process requests
  executor.spin();

  // Shut down the ROS 2 system
  rclcpp::shutdown();
  return 0;
}
