#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include "custom_interfaces/srv/execute_trajectory.hpp"
#include "custom_interfaces/srv/reach_point.hpp"
//#include <kinenik/kinenik_ur.h>
#include <robot_arm_interfaces/srv/ik_arm.hpp>

using namespace std::placeholders;

class CartesianControlNode : public rclcpp::Node
{
public:
    using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
    using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

    CartesianControlNode()
        : Node("cartesian_position_controller")
    {
        // Publisher for direct trajectory testing
        trajectory_publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory_controller/joint_trajectory", 10);

        // Publisher for trajectory completion
        trajectory_complete_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/low_level_control/trajectory_complete", 10);

        // Services
        //reach_point_service_ = this->create_service<custom_interfaces::srv::ReachPoint>(
          //  "/low_level_control/reach_point",
          //  std::bind(&CartesianControlNode::handleReachPoint, this, std::placeholders::_1, std::placeholders::_2));

        execute_trajectory_service_ = this->create_service<custom_interfaces::srv::ExecuteTrajectory>(
            "/low_level_control/execute_trajectory",
            std::bind(&CartesianControlNode::handleExecuteTrajectory, this, std::placeholders::_1, std::placeholders::_2));

        // Action client for FollowJointTrajectory
        action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this, "/right_arm/move");

        client_ik_arm_ = this->create_client<robot_arm_interfaces::srv::IkArm>("/arm_right/ik");

        RCLCPP_INFO(this->get_logger(), "Madar control node initialized.");
    }

private:

  trajectory_msgs::msg::JointTrajectory requestJointPathFromGivenPosePath(geometry_msgs::msg::PoseArray _pose_path)
  {
      auto srv_request = std::make_shared<robot_arm_interfaces::srv::IkArm::Request>();

      // Add the poses of the path (user values):
      for (const auto& pose : _pose_path.poses) {
          srv_request->pose_path.poses.push_back(pose);
      }
      trajectory_msgs::msg::JointTrajectory joint_path_result;
      while (!this->client_ik_arm_->wait_for_service(std::chrono::seconds(5))) {
          if (!rclcpp::ok()) {
              RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the arm_right/ik service. Exiting.");
              return joint_path_result;
          }
          RCLCPP_WARN(this->get_logger(), "Waiting for the arm_right/ik service to be available...");
      }

      auto future_result = this->client_ik_arm_->async_send_request(srv_request);
      RCLCPP_INFO(this->get_logger(),"IK service called");



      // Wait for the result.
      if (rclcpp::spin_until_future_complete(this->shared_from_this(), future_result) == rclcpp::FutureReturnCode::SUCCESS) {
          auto result = future_result.get();
          joint_path_result.points = result->joint_path.points;
          return joint_path_result;

      } else {
          RCLCPP_ERROR(this->get_logger(),"Failed to call IK service");
          return joint_path_result;
      }

    }

    trajectory_msgs::msg::JointTrajectory createTrajectoryMessage(
        const trajectory_msgs::msg::JointTrajectory &joint_path,
        double trajectory_duration,
        double max_velocity)
    {
        // Check for valid inputs
        if (joint_path.points.size() < 2) {
            throw std::invalid_argument("The joint_path must contain at least two points.");
        }
        if (trajectory_duration <= 0) {
            throw std::invalid_argument("The trajectory_duration must be greater than zero.");
        }

        size_t num_points = joint_path.points.size();
        double time_step = trajectory_duration / (num_points - 1); // Time between consecutive points
        trajectory_msgs::msg::JointTrajectory joint_traj;

        for (size_t i = 0; i < num_points; ++i) {
            trajectory_msgs::msg::JointTrajectoryPoint traj_point;
            traj_point.positions = joint_path.points[i].positions;

            if (i > 0) {
                // Calculate velocities as (current_position - previous_position) / time_step
                std::vector<double> velocities;
                for (size_t j = 0; j < joint_path.points[i].positions.size(); ++j) {
                    double current = joint_path.points[i].positions[j];
                    double previous = joint_path.points[i - 1].positions[j];
                    double velocity = (current - previous) / time_step;

                    // Check for velocity limits and warn if exceeded
                    if (std::abs(velocity) > max_velocity) {
                        RCLCPP_WARN(
                            rclcpp::get_logger("trajectory_creator"),
                            "Warning: Joint %zu velocity (%f) exceeds maximum allowed velocity (%f).",
                            j, velocity, max_velocity);
                    }

                    velocities.push_back(velocity);
                }
                traj_point.velocities = velocities;
            } else {
                // For the first point, set velocities to zero
                traj_point.velocities = std::vector<double>(joint_path.points[i].positions.size(), 0.0);
            }

            // Assign timestamp to each point
            //traj_point.time_from_start = rclcpp::Duration::from_seconds(i * time_step);
            joint_traj.points.push_back(traj_point);
        }

        return joint_traj;
    }

    /*void handleReachPoint(
        const std::shared_ptr<custom_interfaces::srv::ReachPoint::Request> request,
        const std::shared_ptr<custom_interfaces::srv::ReachPoint::Response> response)
    {
        double x = request->x;
        double y = request->y; //consider robot being roated wrt the world frame
        double z = request->z;
        double qr = request->qr;
        double qp = request->qp;
        double qy = request->qy;

        std::vector<JointPos> joint_solutions;
        bool success = kinenik_.solveIK(x, y, z, qr, qp, qy, joint_solutions);

        if (!success || joint_solutions.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "No IK solution found. Aborting.");
            response->success = false;
            return;
        }

        auto trajectory_msg = createTrajectoryMessage({joint_solutions[0]}, joint_names_, time_until_reach_point_);

        trajectory_publisher_->publish(trajectory_msg);


        response->success = true;
        RCLCPP_INFO(this->get_logger(), "Published trajectory to reach the target point.");
    }*/

    void handleExecuteTrajectory(
        const std::shared_ptr<custom_interfaces::srv::ExecuteTrajectory::Request> request,
        const std::shared_ptr<custom_interfaces::srv::ExecuteTrajectory::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Entering low_level_control traj execution service...");
        if (!action_client_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting.");
            response->success = false;
            return;
        }

        trajectory_msgs::msg::JointTrajectory joint_path;
        geometry_msgs::msg::PoseArray pose_path;
        pose_path.poses = request->trajectory;
        joint_path = requestJointPathFromGivenPosePath(pose_path);

        auto trajectory_msg = createTrajectoryMessage(joint_path, trajectory_duration_, max_velocity_);

        auto goal_msg = FollowJointTrajectory::Goal();
        goal_msg.trajectory = trajectory_msg;

        //RCLCPP_INFO(this->get_logger(), "Sending trajectory to action server...");

        auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
        send_goal_options.result_callback = [this](const GoalHandleFollowJointTrajectory::WrappedResult &result) {
            std_msgs::msg::Bool msg;
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
            {
                RCLCPP_INFO(rclcpp::get_logger("madar_trajectory_controller"), "Trajectory execution succeeded.");
                msg.data = true; // Trajectory completed successfully
            }
            else
            {
                RCLCPP_ERROR(rclcpp::get_logger("madar_trajectory_controller"), "Trajectory execution failed or canceled.");
                msg.data = false; // Trajectory failed
            }
            trajectory_complete_publisher_->publish(msg); // Publish the completion status
        };

        // Ask for user confirmation
        std::string user_input;
        RCLCPP_INFO(this->get_logger(), "Do you want to execute the trajectory? (y/n): ");
        std::cin >> user_input;

        if (user_input != "y" && user_input != "Y")
        {
            RCLCPP_WARN(this->get_logger(), "Trajectory execution canceled by user.");
            response->success = false;
            return;
        }

        action_client_->async_send_goal(goal_msg, send_goal_options);
        response->success = true;

        // Now convert joint_trajectory into std_msgs::msg::Float64MultiArray[]
        std::vector<std_msgs::msg::Float64MultiArray> joint_trajectory_multiarray;

        for (const auto &point : joint_path.points)
        {
            std_msgs::msg::Float64MultiArray multi_array;

            // Copy joint values into the data field
            for (const auto &joint_value : point.positions)
            {
                multi_array.data.push_back(joint_value);
            }

            joint_trajectory_multiarray.push_back(multi_array);
        }
        response->joint_trajectory = joint_trajectory_multiarray;
    }

    //rclcpp::Service<custom_interfaces::srv::ReachPoint>::SharedPtr reach_point_service_;
    rclcpp::Service<custom_interfaces::srv::ExecuteTrajectory>::SharedPtr execute_trajectory_service_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trajectory_complete_publisher_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_;
    rclcpp::Client<robot_arm_interfaces::srv::IkArm>::SharedPtr client_ik_arm_;

    //KinenikUR kinenik_;
    //std::vector<std::string> joint_names_ = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    //double time_between_traj_points_ = 0.04; //0.004 for DMP-tau=5
    //double time_until_reach_point_ = 3.0;
    double trajectory_duration_ = 3.0;
    double max_velocity_ = 3.14;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CartesianControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
