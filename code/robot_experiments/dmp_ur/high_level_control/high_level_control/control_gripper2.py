import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from rclpy.duration import Duration
from std_srvs.srv import Empty

class GripperControl(Node):
    def __init__(self):
        super().__init__('gripper_control')

        # Create a publisher to the /gripper_controller2/commands topic
        self._publisher = self.create_publisher(Float64MultiArray, '/gripper_controller2/commands', 10)

        # Create service servers for open and close
        self.open_gripper_service = self.create_service(Empty, '/high_level_control/open_gripper', self.open_gripper_callback)
        self.close_gripper_service = self.create_service(Empty, '/high_level_control/close_gripper', self.close_gripper_callback)
        self.opened = 0.0
        self.declare_parameter('closed_gripper',0.6)

    def send_gripper_command(self, positions: list):
        # Create a message of type Float64MultiArray to publish joint positions
        msg = Float64MultiArray()
        msg.data = positions  # The positions for all joints in the gripper

        # Publish the message
        self.get_logger().info(f'Sending gripper command with positions: {positions}')
        self._publisher.publish(msg)

    def open_gripper_callback(self, request, response):
        self.get_logger().info('Open gripper service called.')

        # Open the gripper by setting the main driver joints to 0 (open) and others to their neutral positions
        open_positions = [
            self.opened,  # gripper_right_driver_joint (open)
            self.opened,  # gripper_left_driver_joint (open)
            self.opened,  # gripper_right_spring_link_joint (neutral position)
            self.opened,  # gripper_left_spring_link_joint (neutral position)
            self.opened,  # gripper_right_follower_joint (neutral position)
            self.opened   # gripper_left_follower_joint (neutral position)
        ]
        self.send_gripper_command(open_positions)
        return response

    def close_gripper_callback(self, request, response):
        self.get_logger().info('Close gripper service called.')

        # Close the gripper by setting the main driver joints to 1 (close) and others to their neutral positions
        close_positions = [
            self.get_parameter('closed_gripper').value,  # gripper_right_driver_joint (close)
            self.get_parameter('closed_gripper').value,  # gripper_left_driver_joint (close)
            self.get_parameter('closed_gripper').value,  # gripper_right_spring_link_joint (neutral position)
            self.get_parameter('closed_gripper').value,  # gripper_left_spring_link_joint (neutral position)
            -self.get_parameter('closed_gripper').value,  # gripper_right_follower_joint (neutral position)
            -self.get_parameter('closed_gripper').value   # gripper_left_follower_joint (neutral position)
        ]
        self.send_gripper_command(close_positions)
        return response

def main(args=None):
    rclpy.init(args=args)
    gripper_control = GripperControl()
    rclpy.spin(gripper_control)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
