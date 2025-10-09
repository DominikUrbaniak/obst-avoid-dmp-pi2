#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/bool.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include "custom_interfaces/srv/execute_trajectory.hpp"
#include "custom_interfaces/srv/reach_point.hpp"
#include <kinenik/kinenik_ur.h>

using namespace std::placeholders;

class CartesianControlNode : public rclcpp::Node
{
public:
    using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
    using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

    CartesianControlNode()
        : Node("cartesian_position_controller"), kinenik_("UR5e")
    {
        // Publisher for direct trajectory testing
        trajectory_publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory_controller/joint_trajectory", 10);

        // Publisher for trajectory completion
        trajectory_complete_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/low_level_control/trajectory_complete", 10);

        // Services
        reach_point_service_ = this->create_service<custom_interfaces::srv::ReachPoint>(
            "/low_level_control/reach_point",
            std::bind(&CartesianControlNode::handleReachPoint, this, std::placeholders::_1, std::placeholders::_2));

        execute_trajectory_service_ = this->create_service<custom_interfaces::srv::ExecuteTrajectory>(
            "/low_level_control/execute_trajectory",
            std::bind(&CartesianControlNode::handleExecuteTrajectory, this, std::placeholders::_1, std::placeholders::_2));

        // Action client for FollowJointTrajectory
        action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this, "/joint_trajectory_controller/follow_joint_trajectory");

        RCLCPP_INFO(this->get_logger(), "Cartesian control node initialized.");
    }

private:
    trajectory_msgs::msg::JointTrajectory createTrajectoryMessage(
        const std::vector<JointPos> &joint_trajectory,
        const std::vector<std::string> &joint_names,
        double time_between_points)
    {
        trajectory_msgs::msg::JointTrajectory trajectory_msg;

        // Check if trajectory is valid
        if (joint_trajectory.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Empty joint trajectory.");
            return trajectory_msg;
        }

        // Assign joint names
        trajectory_msg.joint_names = joint_names;

        // Populate trajectory points
        for (size_t i = 0; i < joint_trajectory.size(); ++i)
        {
            trajectory_msgs::msg::JointTrajectoryPoint point;

            // Add joint positions to the point
            point.positions.insert(point.positions.end(), joint_trajectory[i].begin(), joint_trajectory[i].end());

            // Set the time for this point (incremental timing)
            point.time_from_start = rclcpp::Duration::from_seconds(time_between_points * (i + 1));

            trajectory_msg.points.push_back(point);
        }

        return trajectory_msg;
    }

    void handleReachPoint(
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
    }

    void handleExecuteTrajectory(
        const std::shared_ptr<custom_interfaces::srv::ExecuteTrajectory::Request> request,
        const std::shared_ptr<custom_interfaces::srv::ExecuteTrajectory::Response> response)
    {
        //RCLCPP_INFO(this->get_logger(), "Entering low_level_control traj execution service...");
        if (!action_client_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting.");
            response->success = false;
            return;
        }

        std::vector<JointPos> joint_trajectory;

        for (const auto &pose : request->trajectory)
        {
            std::vector<JointPos> joint_solutions;
            bool success = kinenik_.solveIK(
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w, joint_solutions);

            if (!success || joint_solutions.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to solve IK for trajectory point.");
                response->success = false;
                return;
            }

            joint_trajectory.push_back(joint_solutions[0]);
        }

        auto trajectory_msg = createTrajectoryMessage(joint_trajectory, joint_names_, time_between_traj_points_);

        auto goal_msg = FollowJointTrajectory::Goal();
        goal_msg.trajectory = trajectory_msg;

        //RCLCPP_INFO(this->get_logger(), "Sending trajectory to action server...");

        auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
        send_goal_options.result_callback = [this](const GoalHandleFollowJointTrajectory::WrappedResult &result) {
            std_msgs::msg::Bool msg;
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
            {
                RCLCPP_INFO(rclcpp::get_logger("cartesian_position_controller"), "Trajectory execution succeeded.");
                msg.data = true; // Trajectory completed successfully
            }
            else
            {
                RCLCPP_ERROR(rclcpp::get_logger("cartesian_position_controller"), "Trajectory execution failed or canceled.");
                msg.data = false; // Trajectory failed
            }
            trajectory_complete_publisher_->publish(msg); // Publish the completion status
        };

        action_client_->async_send_goal(goal_msg, send_goal_options);
        response->success = true;

        // Now convert joint_trajectory into std_msgs::msg::Float64MultiArray[]
        std::vector<std_msgs::msg::Float64MultiArray> joint_trajectory_multiarray;

        for (const auto &joint_pos : joint_trajectory)
        {
            std_msgs::msg::Float64MultiArray multi_array;

            // Assuming JointPos is a vector or array of joint values, we push each joint value to the multi_array
            for (const auto &joint_value : joint_pos)
            {
                multi_array.data.push_back(joint_value);
            }

            joint_trajectory_multiarray.push_back(multi_array);
        }
        response->joint_trajectory = joint_trajectory_multiarray;
    }

    rclcpp::Service<custom_interfaces::srv::ReachPoint>::SharedPtr reach_point_service_;
    rclcpp::Service<custom_interfaces::srv::ExecuteTrajectory>::SharedPtr execute_trajectory_service_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trajectory_complete_publisher_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_;

    KinenikUR kinenik_;
    std::vector<std::string> joint_names_ = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    double time_between_traj_points_ = 0.04; //0.004 for DMP-tau=5
    double time_until_reach_point_ = 3.0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CartesianControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
