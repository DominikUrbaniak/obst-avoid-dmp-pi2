import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose
import numpy as np
from rcl_interfaces.srv import SetParameters
from custom_interfaces.srv import BoxWallConfig

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class WallConfigurator(Node):
    def __init__(self):
        super().__init__('set_obstacle_config')

        self.client_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv_cb_group = MutuallyExclusiveCallbackGroup()
        # Client to set entity state in Gazebo
        self.gazebo_set_entity_state_client = self.create_client(SetEntityState, '/gazebo_state/set_entity_state',callback_group=self.client_cb_group)
        self.get_logger().info('Waiting for Gazebo set_entity_state service...')

        # Wait for the service to become available
        self.gazebo_set_entity_state_client.wait_for_service()
        self.get_logger().info('Service is now available.')

        # Create a service to configure the wall
        self.wall_service = self.create_service(
            BoxWallConfig, '/high_level_control/set_obstacle_config', self.set_obstacle_config_callback,callback_group=self.srv_cb_group
        )
        self.wall_service_ = self.create_service(
            BoxWallConfig, '/high_level_control/set_obstacle_config_cup', self.set_obstacle_config_cup_callback,callback_group=self.srv_cb_group
        )
        self.wall_service_ = self.create_service(
            BoxWallConfig, '/high_level_control/set_obstacle_config_stack', self.set_obstacle_config_stack_callback,callback_group=self.srv_cb_group
        )
        self.x_init = 0.0
        self.z_init = 0.1
        self.init_cup = [0.4,0.3,0.05] #[0.4,0.3,0.05]
        self.init_cube = [-0.4,0.5,0.02]
        self.range_y_cup = 0.4 #0.4
        self.range_x_cup = 0.0 #0.3


    def gen_wall(self, first_cell_coords, ny, nz):
        """
        Generate wall grid coordinates.

        Args:
            first_cell_coords (tuple): (x, y) coordinates of the first cell.
            ny (int): Number of rows.
            nz (int): Number of columns.

        Returns:
            np.ndarray: Array of shape (ny*nz, 2) containing grid coordinates.
        """
        grid_locations = np.zeros((ny * nz, 2))
        counter = 0
        for i in range(1, ny + 1):
            for j in range(1, nz + 1):
                grid_locations[counter, 0] = first_cell_coords[0] + (i - 1) * 0.05
                grid_locations[counter, 1] = first_cell_coords[1] + (j - 1) * 0.2
                counter += 1
        return grid_locations

    def set_object_pose(self, obj_name, obj_pose):
        """
        Set the pose of an object in Gazebo.

        Args:
            obj_name (str): Name of the object in Gazebo.
            pose (Pose): Desired pose of the object.
        """

        while not self.gazebo_set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set entity state service...')

        set_request = SetEntityState.Request()
        set_request.state.name = obj_name
        set_request.state.pose = obj_pose

        # Call the service
        try:
            set_result = self.gazebo_set_entity_state_client.call(set_request)
            return set_result.success
        except Exception as e:
            self.get_logger().error(f'Error setting pose for {obj_name}: {str(e)}')
            return False


    def set_obstacle_config_callback(self, request, response):
        """
        Configure the wall using the specified number of rows and columns.

        Args:
            first_cell_coords (tuple): (x, y) coordinates of the first cell.
            ny (int): Number of rows.
            nz (int): Number of columns.
        """
        if request.first_x != 0.0:
            grid_locations = self.gen_wall((request.first_x, self.z_init), request.ny, request.nz)
        else:
            random_first_x = 0.4 + 0.2*np.random.rand()
            grid_locations = self.gen_wall((random_first_x, self.z_init), request.ny, request.nz)
        num_objects = min(len(grid_locations), 8)  # Limit to 8 objects

        for i in range(8):
            pose = Pose()

            if i < num_objects:
                pose.position.x = self.x_init
                pose.position.y = grid_locations[i, 0]
                pose.position.z = grid_locations[i, 1]
            else:
                # Place remaining boxes outside the workspace
                pose.position.x = 2.0
                pose.position.y = 2.0
                pose.position.z = 0.0

            pose.orientation.w = 0.707
            pose.orientation.z = 0.707

            obj_name = f'box_{i}'
            if not self.set_object_pose(obj_name, pose):
                response.success = False
                response.message = f'Failed to set pose for {obj_name}'
                return response

        response.success = True
        response.message = 'Wall configured successfully'
        return response

    def set_obstacle_config_cup_callback(self, request, response): #with cup and cube for pick and drop variation
        """
        Configure the wall using the specified number of rows and columns.

        Args:
            first_cell_coords (tuple): (x, y) coordinates of the first cell.
            ny (int): Number of rows.
            nz (int): Number of columns.
        """
        obj_num = 20
        if request.first_x != 0.0:
            grid_locations = self.gen_wall((request.first_x, self.z_init), request.ny, request.nz)
        else:
            random_first_x = 0.3 + 0.2*np.random.rand()
            grid_locations = self.gen_wall((random_first_x, self.z_init), request.ny, request.nz)
        num_objects = min(len(grid_locations), obj_num)  # Limit to 10 objects
        start_pose = request.start_pose

        for i in range(obj_num):
            pose = Pose()

            if i < num_objects:
                pose.position.x = self.x_init
                pose.position.y = grid_locations[i, 0]
                pose.position.z = grid_locations[i, 1]
            else:
                # Place remaining boxes outside the workspace
                pose.position.x = 2.0
                pose.position.y = 2.0
                pose.position.z = 0.0

            pose.orientation.w = 0.707
            pose.orientation.z = 0.707

            obj_name = f'box_{i}'
            if not self.set_object_pose(obj_name, pose):
                response.success = False
                response.message = f'Failed to set pose for {obj_name}'
                return response

        pose = Pose()
        if request.cup_x == 0.0:
            pose.position.x = self.init_cup[0] + self.range_x_cup*np.random.rand() - 0.5*self.range_x_cup
        else:
            pose.position.x = request.cup_x
        if request.cup_y == 0.0:
            pose.position.y = self.init_cup[1] + self.range_y_cup*np.random.rand()
        else:
            pose.position.y = request.cup_y
        pose.position.z = self.init_cup[2]

        obj_name = f'cup'
        if not self.set_object_pose(obj_name, pose):
            response.success = False
            response.message = f'Failed to set pose for {obj_name}'
            return response

        pose = Pose()
        if start_pose.position.x == 0.0 and start_pose.position.y == 0.0:
            pose.position.x = self.init_cube[0]
            pose.position.y = self.init_cube[1]
        else:
            pose.position.x = start_pose.position.x
            pose.position.y = start_pose.position.y
        pose.position.z = self.init_cube[2]
        obj_name = f'cube'
        if not self.set_object_pose(obj_name, pose):
            response.success = False
            response.message = f'Failed to set pose for {obj_name}'
            return response

        response.success = True
        response.message = 'Wall configured successfully'
        return response

    def set_obstacle_config_stack_callback(self, request, response): #with cup and cube for pick and drop variation
        """
        Configure the wall using the specified number of rows and columns.

        Args:
            first_cell_coords (tuple): (x, y) coordinates of the first cell.
            ny (int): Number of rows.
            nz (int): Number of columns.
        """
        grid_locations = self.gen_wall((request.first_x, self.z_init), request.ny, request.nz)
        num_objects = min(len(grid_locations), 8)  # Limit to 8 objects

        for i in range(8):
            pose = Pose()

            if i < num_objects:
                pose.position.x = self.x_init
                pose.position.y = grid_locations[i, 0]
                pose.position.z = grid_locations[i, 1]
            else:
                # Place remaining boxes outside the workspace
                pose.position.x = 2.0
                pose.position.y = 2.0
                pose.position.z = 0.0

            pose.orientation.w = 0.707
            pose.orientation.z = 0.707

            obj_name = f'box_{i}'
            if not self.set_object_pose(obj_name, pose):
                response.success = False
                response.message = f'Failed to set pose for {obj_name}'
                return response

        coords_first_cell = [-0.5,0.5]
        wx = 0.4
        wy = 0.3
        square_size = 0.1
        n_cubes = 3
        cube_locations = self.generate_non_overlapping_squares(coords_first_cell,wx,wy,square_size,n_cubes)
        for i in range(n_cubes-1):
            pose = Pose()

            pose.position.x = cube_locations[i,0]
            pose.position.y = cube_locations[i,1]
            pose.position.z = 0.025

            obj_name = f'cube_{i}'
            if not self.set_object_pose(obj_name, pose):
                response.success = False
                response.message = f'Failed to set pose for {obj_name}'
                return response

        pose = Pose()
        pose.position.x = -cube_locations[2,0]
        pose.position.y = cube_locations[2,1]
        pose.position.z = 0.025

        obj_name = f'cube_{n_cubes-1}'
        if not self.set_object_pose(obj_name, pose):
            response.success = False
            response.message = f'Failed to set pose for {obj_name}'
            return response

        response.success = True
        response.message = 'Wall configured successfully'
        return response

    def generate_non_overlapping_squares(self,first_cell_coords, wx, wy, square_size, num_squares=9):
        squares = [(self.init_cube[0]-first_cell_coords[0],self.init_cube[1]-first_cell_coords[1])]
        attempts = 1000  # Max attempts to place each square

        for _ in range(num_squares):
            for _ in range(attempts):
                # Generate random top-left corner (x, y) such that the square fits within workspace
                x = random.uniform(0, wx - square_size)
                y = random.uniform(0, wy - square_size)

                # Check if the new square overlaps with any existing ones
                no_overlap = True
                for (sx, sy) in squares:
                    if not (x + square_size <= sx or x >= sx + square_size or y + square_size <= sy or y >= sy + square_size):
                        no_overlap = False
                        break

                # If there's no overlap, add the square to the list and break out of the attempt loop
                if no_overlap:
                    squares.append((x, y))
                    break
            else:
                # If we couldn't place a square after max attempts, raise an error
                raise ValueError("Couldn't place all squares without overlap.")
        return np.array(squares) + first_cell_coords


def main(args=None):
    rclpy.init(args=args)
    node = WallConfigurator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')

    executor.shutdown()

if __name__ == '__main__':
    main()
