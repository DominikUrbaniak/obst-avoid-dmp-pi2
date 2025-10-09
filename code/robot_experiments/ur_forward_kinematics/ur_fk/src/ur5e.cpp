#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <ur_fk/msg/joint_poses.hpp>

#include <vector>
#include <cmath>

#include <sensor_msgs/msg/joint_state.hpp>
//#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Quaterniond;
using namespace std::placeholders;

const double d1 = 0.1625;
const double d2 = 0.0;
const double d3 = 0.0;
const double d4 = 0.1333;
const double d5 = 0.0997;
const double d6 = 0.0996;

const double a1 = 0.0;
const double a2 = -0.425;
const double a3 = -0.3922;
const double a4 = 0.0;
const double a5 = 0.0;
const double a6 = 0.0;

const double alpha1 = M_PI_2;
const double alpha2 = 0;
const double alpha3 = 0;
const double alpha4 = M_PI_2;
const double alpha5 = -M_PI_2;
const double alpha6 = 0;

std::vector<Matrix4d> forwardKinematics(const double q1, const double q2, const double q3,
                                        const double q4, const double q5, const double q6) {
    Matrix4d T01, T12, T23, T34, T45, T56;

    T01 << cos(q1), -sin(q1)*cos(alpha1), sin(q1)*sin(alpha1), a1*cos(q1),
           sin(q1), cos(q1)*cos(alpha1), -cos(q1)*sin(alpha1), a1*sin(q1),
           0, sin(alpha1), cos(alpha1), d1,
           0, 0, 0, 1;

    T12 << cos(q2), -sin(q2)*cos(alpha2), sin(q2)*sin(alpha2), a2*cos(q2),
           sin(q2), cos(q2)*cos(alpha2), -cos(q2)*sin(alpha2), a2*sin(q2),
           0, sin(alpha2), cos(alpha2), d2,
           0, 0, 0, 1;

    T23 << cos(q3), -sin(q3)*cos(alpha3), sin(q3)*sin(alpha3), a3*cos(q3),
           sin(q3), cos(q3)*cos(alpha3), -cos(q3)*sin(alpha3), a3*sin(q3),
           0, sin(alpha3), cos(alpha3), d3,
           0, 0, 0, 1;

    T34 << cos(q4), -sin(q4)*cos(alpha4), sin(q4)*sin(alpha4), a4*cos(q4),
           sin(q4), cos(q4)*cos(alpha4), -cos(q4)*sin(alpha4), a4*sin(q4),
           0, sin(alpha4), cos(alpha4), d4,
           0, 0, 0, 1;

    T45 << cos(q5), -sin(q5)*cos(alpha5), sin(q5)*sin(alpha5), a5*cos(q5),
           sin(q5), cos(q5)*cos(alpha5), -cos(q5)*sin(alpha5), a5*sin(q5),
           0, sin(alpha5), cos(alpha5), d5,
           0, 0, 0, 1;

    T56 << cos(q6), -sin(q6)*cos(alpha6), sin(q6)*sin(alpha6), a6*cos(q6),
           sin(q6), cos(q6)*cos(alpha6), -cos(q6)*sin(alpha6), a6*sin(q6),
           0, sin(alpha6), cos(alpha6), d6,
           0, 0, 0, 1;

    // Compute cumulative transformations
    Matrix4d T02 = T01 * T12;
    Matrix4d T03 = T02 * T23;
    Matrix4d T04 = T03 * T34;
    Matrix4d T05 = T04 * T45;
    Matrix4d T06 = T05 * T56;

    // Store all transformations in a vector
    std::vector<Matrix4d> joint_poses = {T01, T02, T03, T04, T05, T06};

    return joint_poses;
}

geometry_msgs::msg::Pose eigenToRosPose(const Matrix4d &eigen_matrix)
{
    geometry_msgs::msg::Pose pose;

    pose.position.x = eigen_matrix(0, 3);
    pose.position.y = eigen_matrix(1, 3);
    pose.position.z = eigen_matrix(2, 3);
    Eigen::Matrix3d rotation_matrix = eigen_matrix.block<3, 3>(0, 0);
    Eigen::Quaterniond quaternion(rotation_matrix);
    pose.orientation.x = quaternion.x();
    pose.orientation.y = quaternion.y();
    pose.orientation.z = quaternion.z();
    pose.orientation.w = quaternion.w();

    return pose;
}

class UR5eFKPublisher : public rclcpp::Node {
public:
    UR5eFKPublisher() : Node("ur5e_fk_publisher"), initialized_(false) {
        all_pose_publisher_ = this->create_publisher<ur_fk::msg::JointPoses>("/ur5e/all_joint_poses", 10);
        //pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("/ur5e/end_effector_pose", 10);
        //euler_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/ur5e/end_effector_euler", 10);

        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states",
            10,
            std::bind(&UR5eFKPublisher::joint_states_callback, this, _1)
        );
    }

private:
    // Initialize the mapping from joint names to indices once
    void initialize_joint_mapping(const sensor_msgs::msg::JointState &msg) {
        joint_indices_ = {
            {"shoulder_pan_joint", 0},
            {"shoulder_lift_joint", 1},
            {"elbow_joint", 2},
            {"wrist_1_joint", 3},
            {"wrist_2_joint", 4},
            {"wrist_3_joint", 5},
        };

        // Populate joint_name_to_index_ based on the received joint state message
        for (size_t i = 0; i < msg.name.size(); ++i) {
            const std::string& joint_name = msg.name[i];
            if (joint_indices_.find(joint_name) != joint_indices_.end()) {
                joint_name_to_index_[joint_name] = i;
            }
        }
    }

    void joint_states_callback(const sensor_msgs::msg::JointState & msg)
    {
        if (!initialized_) {
              initialize_joint_mapping(msg);
              initialized_ = true;
          }

        double q1 = msg.position[joint_name_to_index_["shoulder_pan_joint"]];
        double q2 = msg.position[joint_name_to_index_["shoulder_lift_joint"]];
        double q3 = msg.position[joint_name_to_index_["elbow_joint"]];
        double q4 = msg.position[joint_name_to_index_["wrist_1_joint"]];
        double q5 = msg.position[joint_name_to_index_["wrist_2_joint"]];
        double q6 = msg.position[joint_name_to_index_["wrist_3_joint"]];

        // Extract rotation matrix and compute quaternion
        //Eigen::Matrix3d rotation_matrix = T.block<3, 3>(0, 0);
        //Quaterniond quaternion(rotation_matrix);

        // Compute Euler angles (roll, pitch, yaw)
        //Vector3d euler_angles = rotation_matrix.eulerAngles(0, 1, 2);

        // Get forward kinematics for all joints
        auto joint_poses = forwardKinematics(q1, q2, q3, q4, q5, q6);

        // Joint names corresponding to transformations
        std::vector<std::string> joint_names = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

        // Create message
        ur_fk::msg::JointPoses poses_msg;
        poses_msg.header.stamp = this->now();
        poses_msg.header.frame_id = "base_link";
        poses_msg.joint_names = joint_names;

        // Convert Eigen matrices to ROS poses and add to the message
        for (const auto &pose_matrix : joint_poses)
        {
            poses_msg.poses.push_back(eigenToRosPose(pose_matrix));
        }

        // Publish the message
        all_pose_publisher_->publish(poses_msg);

        // Publish end-effector pose (last pose)
        //pose_publisher_->publish(eigenToRosPose(joint_poses.back()));

        // Publish Euler angles
        //std_msgs::msg::Float64MultiArray euler_msg;
        //euler_msg.data = {euler_angles.x(), euler_angles.y(), euler_angles.z()};

        //euler_publisher_->publish(euler_msg);

        //RCLCPP_INFO(this->get_logger(), "Position: [%.2f, %.2f, %.2f]", position.x(), position.y(), position.z());
        //RCLCPP_INFO(this->get_logger(), "Orientation (quaternion): [%.2f, %.2f, %.2f, %.2f]", quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
        //RCLCPP_INFO(this->get_logger(), "Orientation (Euler): [%.2f, %.2f, %.2f]", euler_angles.x(), euler_angles.y(), euler_angles.z());
    }

    //rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;
    //rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr euler_publisher_;
    rclcpp::Publisher<ur_fk::msg::JointPoses>::SharedPtr all_pose_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unordered_map<std::string, int> joint_indices_;
    std::unordered_map<std::string, int> joint_name_to_index_;
    bool initialized_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UR5eFKPublisher>());
    rclcpp::shutdown();
    return 0;
}
