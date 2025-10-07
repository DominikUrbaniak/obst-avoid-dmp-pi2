import os
import sys
import rclpy
from rclpy.node import Node
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
#import cv2
import numpy as np
import time

from geometry_msgs.msg import Pose, Vector3
from std_srvs.srv import Empty

from pympler.asizeof import asizeof
from sklearn.cluster import DBSCAN
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from custom_interfaces.srv import GetAvoidanceParams, GetBoundingBoxParams

from scipy.spatial import distance #using distance.cdist for 1-param (circular) avoidance
from ur_fk.msg import JointPoses

from scipy.spatial.transform import Rotation as R

class ObstacleDetectionNode(Node):
    def __init__(self):
        super().__init__('obstacle_detection_node')
        self.declare_parameter('visualize_obst_view',False)
        self.declare_parameter('visualize_obst_view_all',False)
        self.declare_parameter('obst_limit_y',0.25)
        #self.cv_bridge = CvBridge()
        #self.cameraMatrix = np.array([[554.254691191187,0.0,320.5],[0.0,554.254691191187,240.5],[0,0,1]]) #simulated camera libgazebo_ros_camera: /camera/camera_info
        #self.cameraMatrix = self.cameraMatrix.astype('float32')
        #self.distortionCoeffs = np.array([0.0,0.0,0.0,0.0]) #k1,k2,p1,p2
        #self.distortionCoeffs = self.distortionCoeffs.astype('float32')
        self.camera_pos = np.array([0.0,0.8,1.1])
        self.camera_rot = np.array([-0.1*np.pi,np.pi,np.pi])
        self.git_dir = 'src/dmp_ur/'
        self.pcd_dir = 'src/dmp_ur/docs/pcds/'
        self.base_dir = f"{self.git_dir}docs/experiments/obst_avoid_1_param/"
        if not os.path.exists(self.pcd_dir):
            os.makedirs(self.pcd_dir)
        self.obst_view_dir = ''
        self.file_note = ""
        self.rc = 0

        self.service = self.create_service(GetAvoidanceParams, 'image_processing/get_avoidance_params', self.get_avoidance_params)
        self.service_ = self.create_service(GetBoundingBoxParams, 'image_processing/get_bounding_box_params', self.get_bounding_box_params)
        self.service_save_pcd = self.create_service(Empty, 'image_processing/save_pcd', self.save_pcd)
        self.service_vis_pcd = self.create_service(Empty, 'image_processing/vis_pcd', self.visualize_pcd)

        #self.get_logger().info(f'Starting measurement with rc?->{self.rc}, with qos_profile: {self.qos_profile}, logging: {self.logging}, measurement duration: {self.communication_duration}')

        if self.rc:
            self.pc_subscriber = self.create_subscription(
                PointCloud2,
                '/stereo/points2',
                self.point_cloud_callback,
                10
            )

        else:
            self.pc_subscriber = self.create_subscription(
                PointCloud2,
                '/camera/points',
                self.point_cloud_callback,
                10
            )

        self.o3d_cloud_all = None
        self.o3d_cloud_obst_obj = None
        self.point_cloud_data_eef_orig = None
        self.point_cloud_data_eef = None
        self.point_cloud_data_all = None
        self.point_cloud_data = None
        self.point_cloud_data_wo_floor = None
        self.point_cloud_data_wo_floor_w_obj = None
        self.min_p_z_all = None
        self.max_p_z_all = None
        self.eef_position = None
        self.obst_limits_x = [-0.1,0.1]
        #self.obst_limit_y = 0.25
        self.obj_dims = [0.08,0.08,0.12]
        #self.obj_dims = [0.05,0.05,0.05]
        self.pos_obj = np.zeros(3)
        self.joint0_quat = None
        self.link0_offset = 0.18
        self.link0_offset_angle = np.pi/2

        self.robot_link_radius = 0.1 #remove points that are within that radius between the link
        self.eef_pose = None
        self.joint_poses = None
        self.joint_points = None
        self.obst_bbs = None
        self.create_subscription(JointPoses, '/ur5e/all_joint_poses', self.eef_pose_callback, 10)

        self.get_logger().info(f'Started obstacle_detection node')

    def eef_pose_callback(self,msg):
        self.joint_poses = msg.poses
        self.eef_pose = msg.poses[-1] #get the wrist3 (eef) pose
        self.joint0_quat = msg.poses[0].orientation

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to NumPy array
        self.point_cloud_data_eef_orig = np.array(pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))
        #self.o3d_point_cloud = np.array(pc2.read_points_list(msg, field_names=("x", "y"), skip_nans=True))

    def visualize_pcd(self,req,res):
        self.point_cloud_data_eef = self.transform_point_cloud(self.point_cloud_data_eef_orig,self.camera_pos,self.camera_rot)
        o3d_cloud_all = o3d.geometry.PointCloud()
        o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef)
        o3d.visualization.draw_geometries([o3d_cloud_all])

        return res

    def save_pcd(self,req,res):
        s = f"{self.pcd_dir}pcd_rc_{self.rc}_{int(time.time())}"
        o3d_cloud_all = o3d.geometry.PointCloud()
        o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef_orig)
        o3d.io.write_point_cloud(f'{s}.pcd',o3d_cloud_all)
        self.point_cloud_data_eef = self.transform_point_cloud(self.point_cloud_data_eef_orig,self.camera_pos,self.camera_rot)
        np.savez(f'{s}.npz', point_cloud_all=self.point_cloud_data_eef_orig,point_cloud_transformed=self.point_cloud_data_eef)
        self.get_logger().info(f'saved point cloud at {s}')
        return res

    def create_obb(self,point1, point2,angle):
        # Calculate center and line vector
        center = (point1 + point2) / 2

        #center[2] = (self.max_p_z_all + self.min_p_z_all) / 2 + 0.002 #exlude the surface
        #height = 2.0
        #center[2] = 1.01 #
        line_vector = point2 - point1
        #line_vector[2] = 0.0 #only rotate around Z
        length = np.linalg.norm(line_vector)
        line_vector /= length  # Normalize the line_vector

        # Define the rotation matrix aligning the bounding box's X-axis with line_vector
        x_axis = line_vector
        # Choose an arbitrary vector not aligned with line_vector to compute the Y-axis
        arbitrary_vec = np.array([0, 0, 1]) if np.abs(line_vector[2]) < 0.99 else np.array([0, 1, 0])
        y_axis = np.cross(x_axis, arbitrary_vec)
        y_axis /= np.linalg.norm(y_axis)
        #z_axis = abs(np.cross(x_axis, y_axis)) #must be positive
        z_axis = np.cross(x_axis, y_axis)


        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Define the extent of the bounding box
        extent = 0.8*np.ones(3)  # Set 2.0 in each direction
        # Create the Oriented Bounding Box
        center_obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
        #self.get_logger().info(f'init_point: {point1}, end_point: {point2}, midpoint: {midpoint}, point: {point}, proj_to_line: {proj_to_line}, proj_to_width: {proj_to_width} ')
        return center_obb, rotation_matrix, z_axis


    def create_3d_crop(self,points,point1, point2, width):
        """Crop points within a 3D bounding box defined by point1, point2, and width.

        Args:
            point1 (np.array): Start point [x, y, z].
            point2 (np.array): End point [x, y, z].
            width (float): Width of the bounding box around the line connecting point1 and point2.
            points (np.array): 3D point cloud array (N, 3).

        Returns:
            np.array: Points within the bounding box.
        """
        # Calculate bounding box limits based on line segment
        center_line = point2 - point1
        line_length = np.linalg.norm(center_line)
        unit_line = center_line / line_length if line_length > 0 else center_line

        # Define perpendicular vectors for box width
        perp_vector1 = np.cross(unit_line, np.array([1, 0, 0]))
        if np.linalg.norm(perp_vector1) < 1e-6:
            perp_vector1 = np.cross(unit_line, np.array([0, 1, 0]))
        perp_vector2 = np.cross(unit_line, perp_vector1)

        # Normalize and scale for width
        perp_vector1 = perp_vector1 / np.linalg.norm(perp_vector1) * width / 2
        perp_vector2 = perp_vector2 / np.linalg.norm(perp_vector2) * width / 2

        # Calculate the 8 corners of the bounding box
        corners = [
            point1 + perp_vector1 + perp_vector2,
            point1 + perp_vector1 - perp_vector2,
            point1 - perp_vector1 + perp_vector2,
            point1 - perp_vector1 - perp_vector2,
            point2 + perp_vector1 + perp_vector2,
            point2 + perp_vector1 - perp_vector2,
            point2 - perp_vector1 + perp_vector2,
            point2 - perp_vector1 - perp_vector2,
        ]

        self.get_logger().info(f'cropped box corners: {corners}')

        # Calculate min and max bounds for each axis
        min_bounds = np.min(corners, axis=0)
        max_bounds = np.max(corners, axis=0)

        # Filter points within the bounding box
        mask = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)
        cropped_points = points[mask]
        return cropped_points

    def get_avoidance_params(self, req, res):
        start_time0 = time.time()
        self.get_logger().info(f'communication time from control: {start_time0 - req.time_stamp}')
        computation_times = {}
        self.obst_view_dir = req.dir_path
        start_time = time.time()
        # Voxel downsample (choose a voxel size based on your data scale, e.g., 0.01 meters)
        o3d_cloud_all = o3d.geometry.PointCloud()
        o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef_orig)

        voxel_size = 0.01
        o3d_cloud = o3d_cloud_all.voxel_down_sample(voxel_size)
        self.point_cloud_data = np.asarray(o3d_cloud.points)
        computation_times['part0_pcd_down_sample'] = time.time() - start_time
        start_time = time.time()

        #self.obst_view_dir = req.dir_path + "obst_view/"
        n_solutions = req.n_solutions
        self.point_cloud_data_eef = self.transform_point_cloud(self.point_cloud_data,self.camera_pos,self.camera_rot)
        self.o3d_cloud_all = o3d.geometry.PointCloud()
        self.o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef)
        self.min_p_z_all = np.min(self.point_cloud_data_eef[:,2])
        self.max_p_z_all = np.max(self.point_cloud_data_eef[:,2])
        computation_times['part1_transform_pcd'] = time.time() - start_time
        start_time = time.time()
        #remove gripper from the point cloud7
        eef_pose = req.pose_0
        goal_pose = req.pose_goal
        obj_dims = req.obj_dims
        obst_avoid_only = True
        if goal_pose.position.x == 0.0 and goal_pose.position.y == 0.0:
            obst_avoid_only = False

        self.eef_position = np.array([-eef_pose.position.x,-eef_pose.position.y,eef_pose.position.z]) #from camera perspective
        goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z])
        action = 'place'
        if self.eef_position[0] > 0.0:
            action = 'pick'
        self.get_logger().info(f"Action: {action}")
        #eef_height = 0.3
        #eef_length = 0.17
        #eef_width = 0.09
        #min_bound = [self.eef_position[0]-0.5*eef_length,self.eef_position[1]-0.5*eef_width,0.0]
        #max_bound = [self.eef_position[0]+0.5*eef_length,self.eef_position[1]+0.5*eef_width,eef_height]
        #create bounding box
        # Apply the bounding box filter
        #in_bounds = np.all((self.point_cloud_data_eef >= min_bound) & (self.point_cloud_data_eef <= max_bound), axis=1)
        self.point_cloud_data_all,self.joint_points = self.remove_robot_links_from_point_cloud(self.point_cloud_data_eef,self.joint_poses) #[~in_bounds]  # Keep points outside the bounding box
        computation_times['part2_remove_robot_pcd'] = time.time() - start_time
        start_time = time.time()

        #obst_limit_y = np.maximum(self.obst_limit_y,self.eef_position[1]/2)

        self.point_cloud_data_wo_floor = self.point_cloud_data_all[
            (self.point_cloud_data_all[:, 2] > 0.01) &
            (self.point_cloud_data_all[:, 0] > self.obst_limits_x[0]) &
            (self.point_cloud_data_all[:, 0] < self.obst_limits_x[1]) &
            (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
        ]
        if action == 'pick': #when eef in on one side, increase the detection space to the other side (also depends on whether to pick or place an object)
            self.point_cloud_data_wo_floor_w_obj = self.point_cloud_data_all[
                (self.point_cloud_data_all[:, 2] > 0.01) &
                (self.point_cloud_data_all[:, 0] < self.obst_limits_x[1]) &
                (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
            ] #remove points of the floor
        else:
            self.point_cloud_data_wo_floor_w_obj = self.point_cloud_data_all[
                (self.point_cloud_data_all[:, 2] > 0.01) &
                (self.point_cloud_data_all[:, 0] > self.obst_limits_x[0]) &
                (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
            ]
        self.o3d_cloud_obst_obj = o3d.geometry.PointCloud()
        self.o3d_cloud_obst_obj.points = o3d.utility.Vector3dVector(self.point_cloud_data_wo_floor_w_obj)
        # Set all colors to black (0, 0, 0)
        black_colors = np.zeros_like(self.point_cloud_data_wo_floor_w_obj)  # same shape as points
        self.o3d_cloud_obst_obj.colors = o3d.utility.Vector3dVector(black_colors)
        computation_times['part3_remove_floor'] = time.time() - start_time
        start_time = time.time()
        #o3d_cloud_test = o3d.geometry.PointCloud()
        #o3d_cloud_test.points = o3d.utility.Vector3dVector(self.point_cloud_data_wo_floor_w_obj)
        #o3d.visualization.draw_geometries([o3d_cloud_test])
        obj_pose = Pose()
        if len(obj_dims) > 2:
            obj_pose,self.pos_obj,self.obj_bb, obj_pose_center, obj_size = self.detect_obj(self.point_cloud_data_wo_floor_w_obj,obj_dims)
            if np.count_nonzero(self.pos_obj) > 0:
                if action == 'pick':
                    goal_position = np.array([obj_pose.position.x,obj_pose.position.y,np.maximum(0,obj_pose.position.z-obj_dims[2])]) #grasp location
                else:
                    goal_position = np.array([obj_pose.position.x,obj_pose.position.y,obj_pose.position.z])
                obj_pose.position.z = goal_position[2]
            else:
                self.get_logger().warn(f'object not found, got to default goal position')
                #obj_pose.position.x
        else:
            self.get_logger().warn(f'no 3D object dims given (len(obj_dims) <= 2), going to default goal')

        self.get_logger().info(f'obj pose: {obj_pose}')

        computation_times['part4_detect_goal_object'] = time.time() - start_time

        start_time = time.time()
        obstacle_poses,obstacle_sizes,self.obst_bbs = self.detect_bounding_boxes()

        obstacle_poses.append(obj_pose_center)
        obstacle_sizes.append(obj_size)
        self.get_logger().info(f'obstacle poses: {obstacle_poses}, obstacle sizes: {obstacle_sizes}')

        computation_times['part5_detect_obstacle_bounding_boxes'] = time.time() - start_time

        start_time = time.time()
        # Define tag centers and cropping width
        ins1,ins2,ins3,angles,rotation_matrix,min_max_xyz = self.detect_obstacles(self.eef_position, goal_position, n_solutions)

        computation_times['part6_detect_obstacle_parameters'] = time.time() - start_time

        start_time = time.time()

        res.poses = [obj_pose]
        res.ins1 = ins1
        res.ins2 = ins2
        res.ins3 = ins3

        res.angles = angles

        res.obstacle_poses = obstacle_poses
        res.obstacle_sizes = obstacle_sizes
        # Publish images if necessary
        #self.publish_obstacle_views(obst_view_init_crop, obst_view_main_crop, obst_view_end_crop)
        #self.obst_view_counter += 1
        s = self.obst_view_dir + 'point_cloud_data.npz'
        np.savez(s, point_cloud_with_eef=self.point_cloud_data_eef, point_cloud_wo_floor=self.point_cloud_data_wo_floor,point_cloud_data_wo_floor_w_obj=self.point_cloud_data_wo_floor_w_obj,eef_position=self.eef_position,goal_position=goal_position,
                obstacle_poses=obstacle_poses, obstacle_sizes=obstacle_sizes,min_max_xyz_=min_max_xyz,rotation_matrix=rotation_matrix,obst_limit_y=self.get_parameter('obst_limit_y').value,
                voxel_size=voxel_size,computation_times=computation_times,obj_pos=self.pos_obj,obst_avoid_only=obst_avoid_only,joint_poses=self.joint_poses)
        res.success = True
        self.get_logger().info(f'computation time for logging: {time.time() - start_time}, complete detection computation time: {time.time()-start_time0}')
        res.time_stamp_start = start_time0
        res.time_stamp_end = time.time()
        return res

    def get_bounding_box_params(self, req, res):
        start_time0 = time.time()
        success = True
        computation_times = {}
        self.obst_view_dir = req.dir_path
        start_time = time.time()

        # Voxel downsample (choose a voxel size based on your data scale, e.g., 0.01 meters)
        o3d_cloud_all = o3d.geometry.PointCloud()
        o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef_orig)

        voxel_size = 0.01
        o3d_cloud = o3d_cloud_all.voxel_down_sample(voxel_size)
        self.point_cloud_data = np.asarray(o3d_cloud.points)
        computation_times['part0_pcd_down_sample'] = time.time() - start_time
        start_time = time.time()
        #self.obst_view_dir = req.dir_path + "obst_view/"
        n_solutions = req.n_solutions
        self.point_cloud_data_eef = self.transform_point_cloud(self.point_cloud_data,self.camera_pos,self.camera_rot)
        self.o3d_cloud_all = o3d.geometry.PointCloud()
        self.o3d_cloud_all.points = o3d.utility.Vector3dVector(self.point_cloud_data_eef)
        self.min_p_z_all = np.min(self.point_cloud_data_eef[:,2])
        self.max_p_z_all = np.max(self.point_cloud_data_eef[:,2])
        computation_times['part1_transform_pcd'] = time.time() - start_time
        start_time = time.time()
        #remove gripper from the point cloud7
        eef_pose = req.pose_0
        goal_pose = req.pose_goal
        obj_dims = req.obj_dims
        obst_avoid_only = True
        if goal_pose.position.x == 0.0 and goal_pose.position.y == 0.0:
            obst_avoid_only = False

        self.eef_position = np.array([-eef_pose.position.x,-eef_pose.position.y,eef_pose.position.z]) #from camera perspective
        #self.get_logger().info(f"Initial position: {self.eef_position}")
        goal_position = np.array([goal_pose.position.x,goal_pose.position.y,goal_pose.position.z])
        action = 'place'
        #obst_center_x = (np.sum(self.obst_limits_x)/2)
        if self.eef_position[0] > 0.0:
            action = 'pick'
        self.get_logger().info(f"Action: {action}")
        #eef_height = 0.3
        #eef_length = 0.17
        #eef_width = 0.09
        #min_bound = [self.eef_position[0]-0.5*eef_length,self.eef_position[1]-0.5*eef_width,0.0]
        #max_bound = [self.eef_position[0]+0.5*eef_length,self.eef_position[1]+0.5*eef_width,eef_height]
        #create bounding box
        # Apply the bounding box filter
        #in_bounds = np.all((self.point_cloud_data_eef >= min_bound) & (self.point_cloud_data_eef <= max_bound), axis=1)
        self.point_cloud_data_all,self.joint_points = self.remove_robot_links_from_point_cloud(self.point_cloud_data_eef,self.joint_poses)
        computation_times['part2_remove_robot_pcd'] = time.time() - start_time
        start_time = time.time()

        self.point_cloud_data_wo_floor = self.point_cloud_data_all[
            (self.point_cloud_data_all[:, 2] > 0.01) &
            (self.point_cloud_data_all[:, 0] > self.obst_limits_x[0]) &
            (self.point_cloud_data_all[:, 0] < self.obst_limits_x[1]) &
            (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
        ]
        if action == 'pick': #when eef in on one side, increase the detection space to the other side (also depends on whether to pick or place an object)
            self.point_cloud_data_wo_floor_w_obj = self.point_cloud_data_all[
                (self.point_cloud_data_all[:, 2] > 0.01) &
                (self.point_cloud_data_all[:, 0] < self.obst_limits_x[1]) &
                (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
            ] #remove points of the floor
        else:
            self.point_cloud_data_wo_floor_w_obj = self.point_cloud_data_all[
                (self.point_cloud_data_all[:, 2] > 0.01) &
                (self.point_cloud_data_all[:, 0] > self.obst_limits_x[0]) &
                (np.absolute(self.point_cloud_data_all[:, 1]) > self.get_parameter('obst_limit_y').value)
            ]

        computation_times['part3_remove_floor'] = time.time() - start_time
        start_time = time.time()
        #o3d_cloud_test = o3d.geometry.PointCloud()
        #o3d_cloud_test.points = o3d.utility.Vector3dVector(self.point_cloud_data_goal_region)
        #o3d.visualization.draw_geometries([o3d_cloud_test])
        obj_pose = Pose()
        if len(obj_dims) > 2:
            obj_pose,self.pos_obj,self.obj_bb, obj_pose_center, obj_size = self.detect_obj(self.point_cloud_data_wo_floor_w_obj,obj_dims)
            if np.count_nonzero(self.pos_obj) > 0:
                if action == 'pick':
                    goal_position = np.array([obj_pose.position.x,obj_pose.position.y,np.maximum(0,obj_pose.position.z-obj_dims[2])]) #grasp location
                else:
                    goal_position = np.array([obj_pose.position.x,obj_pose.position.y,obj_pose.position.z])
                obj_pose.position.z = goal_position[2]
            else:
                self.get_logger().warn(f'object not found, abort')
                success = False
                #obj_pose.position.x
        else:
            self.get_logger().warn(f'no 3D object dims given (len(obj_dims) <= 2), going to default goal')

        self.get_logger().info(f'obj pose: {obj_pose}')

        computation_times['part4_detect_goal_object'] = time.time() - start_time
        start_time = time.time()


        #if self.get_parameter('visualize_gripper_box').value:
        #    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        #    o3d.visualization.draw_geometries([bounding_box,o3d_cloud])

        # Define tag centers and cropping width
        #start_time = time.time()
        obstacle_poses,obstacle_sizes,self.obst_obbs = self.detect_bounding_boxes()

        obstacle_poses.append(obj_pose_center)
        obstacle_sizes.append(obj_size)
        self.get_logger().info(f'obstacle poses: {obstacle_poses}, obstacle sizes: {obstacle_sizes}')

        computation_times['part5_detect_obstacle_bounding_boxes'] = time.time() - start_time
        start_time = time.time()

        res.poses = [obj_pose]
        res.obstacle_poses = obstacle_poses
        res.obstacle_sizes = obstacle_sizes
        # Publish images if necessary
        #self.publish_obstacle_views(obst_view_init_crop, obst_view_main_crop, obst_view_end_crop)
        #self.obst_view_counter += 1
        s = self.obst_view_dir + 'point_cloud_data.npz'
        if success:
            np.savez(s, point_cloud_with_eef=self.point_cloud_data_eef, point_cloud_wo_floor=self.point_cloud_data_wo_floor,point_cloud_data_wo_floor_w_obj=self.point_cloud_data_wo_floor_w_obj,eef_position=self.eef_position,goal_position=goal_position,
                    obstacle_poses=obstacle_poses, obstacle_sizes=obstacle_sizes,obst_limit_y=self.get_parameter('obst_limit_y').value,
                    voxel_size=voxel_size,computation_times=computation_times,obj_pos=self.pos_obj,obst_avoid_only=obst_avoid_only,joint_poses=self.joint_poses)
        res.success = success
        self.ready_to_start = False
        self.get_logger().info(f'computation time for logging: {time.time() - start_time}, complete detection computation time: {time.time()-start_time0}')
        res.time_stamp_start = start_time0
        res.time_stamp_end = time.time()
        return res

    def detect_obj(self, points, obj_dims):
        points_2d = points[:, :2]
        points_z = points[:, 2]
        # Cluster objects using DBSCAN
        clustering = DBSCAN(eps=0.02, min_samples=10).fit(points_2d)
        labels = clustering.labels_

        clusters = [points_2d[labels == i] for i in range(max(labels) + 1)]
        clusters_z = [points[labels == i] for i in range(max(labels) + 1)]
        #self.get_logger().info(f'Clusters: {clusters[0]}, {clusters_z[0]}') #
        # Match clusters to obj dimensions
        obj_pose = Pose()
        obj_pose_center = Pose()
        obj_size = Vector3()
        obj_cluster = None
        pos_obj = np.zeros(3)
        obj_dims = np.array(obj_dims)
        obj_bb = None
        i=0
        for cluster in clusters:
            i+=1
            # Compute the bounding box dimensions
            min_coords = np.min(cluster, axis=0)
            max_coords = np.max(cluster, axis=0)
            cluster_dims = max_coords - min_coords

            # Match dimensions to obj
            if np.allclose(cluster_dims, obj_dims[:2], atol=0.02):
                obj_cluster = cluster
                obj_cluster_z = clusters_z[i-1]
                # Convert to Open3D point cloud for AABB
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obj_cluster_z)  # Use full 3D points
                obj_bb = pcd.get_axis_aligned_bounding_box()
                obj_bb.color = (0,1,0)
                self.get_logger().info(f'Found obj cluster')
            self.get_logger().info(f'Clusters #{i}, dims: {cluster_dims}, bb: {obj_bb}')

        if obj_cluster is not None:
            min_x = np.min(obj_cluster[:,0])
            max_x = np.max(obj_cluster[:,0])
            min_y = np.min(obj_cluster[:,1])
            max_y = np.max(obj_cluster[:,1])
            obj_pos = np.array([(min_x+max_x)/2,(min_y+max_y)/2])
            pos_obj=np.array([obj_pos[0],obj_pos[1],np.max(obj_cluster_z[:,2])])
            #obj_pose = Pose()
            obj_pose.position.x = obj_pos[0]
            obj_pose.position.y = obj_pos[1]
            obj_pose.position.z = np.max(obj_cluster_z[:,2])

            #obst_pose_center = Pose()
            obj_pose_center.position.x = obj_bb.get_center()[0]
            obj_pose_center.position.y = obj_bb.get_center()[1]
            obj_pose_center.position.z = obj_bb.get_center()[2]
            obj_pose_center.orientation.w = 1.0

            #obj_size = Vector3()
            obj_size.x = obj_bb.get_extent()[0]
            obj_size.y = obj_bb.get_extent()[1]
            obj_size.z = obj_bb.get_extent()[2]

        return obj_pose, pos_obj, obj_bb, obj_pose_center, obj_size

    def detect_bounding_boxes(self):

        obstacle_poses = []
        obstacle_sizes = []
        aabb_list = []
        if self.point_cloud_data_wo_floor.size != 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data_wo_floor)  # pc: Nx3 numpy array
            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=2))

            for i in np.unique(labels):
                if i == -1: continue
                cluster = pcd.select_by_index(np.where(labels == i)[0])
                aabb = cluster.get_axis_aligned_bounding_box()
                aabb.color = (0, 0, 1)

                obst_pose = Pose()
                obst_pose.position.x = aabb.get_center()[0]
                obst_pose.position.y = aabb.get_center()[1]
                obst_pose.position.z = aabb.get_center()[2]
                obst_pose.orientation.w = 1.0

                obst_size = Vector3()
                obst_size.x = aabb.get_extent()[0]
                obst_size.y = aabb.get_extent()[1]
                obst_size.z = aabb.get_extent()[2]

                obstacle_poses.append(obst_pose)
                obstacle_sizes.append(obst_size)
                self.get_logger().info(f'Obstacle cluster #{i} with size: {obst_size} at {obst_pose.position}')
                aabb_list.append(aabb)

        return obstacle_poses, obstacle_sizes, aabb_list

    import open3d as o3d


    def detect_obbs(self):
        obstacle_poses = []
        obstacle_sizes = []
        obb_list = []

        if self.point_cloud_data_wo_floor.size != 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data_wo_floor)
            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=2))

            for i in np.unique(labels):
                if i == -1:
                    continue

                cluster = pcd.select_by_index(np.where(labels == i)[0])
                obb = cluster.get_oriented_bounding_box()
                obb.color = (0, 0, 1)

                # Pose
                center = obb.center
                rotation_matrix = obb.R
                quat = R.from_matrix(rotation_matrix.copy()).as_quat()  # Returns [x, y, z, w]

                obst_pose = Pose()
                obst_pose.position.x = center[0]
                obst_pose.position.y = center[1]
                obst_pose.position.z = center[2]
                obst_pose.orientation.x = quat[0]
                obst_pose.orientation.y = quat[1]
                obst_pose.orientation.z = quat[2]
                obst_pose.orientation.w = quat[3]

                # Size
                extent = obb.extent
                obst_size = Vector3()
                obst_size.x = extent[0]
                obst_size.y = extent[1]
                obst_size.z = extent[2]

                obstacle_poses.append(obst_pose)
                obstacle_sizes.append(obst_size)
                obb_list.append(obb)

                self.get_logger().info(f'Obstacle cluster #{i} with size: {obst_size} at {obst_pose.position}')

            # Visualize (optional)
            #o3d.visualization.draw_geometries([pcd] + obb_list)

        return obstacle_poses, obstacle_sizes, obb_list



    def detect_obstacles(self, point_0, point_goal, n_solutions):

        # Create spheres for start and goal points
        point1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point1.translate(point_0)
        point1.paint_uniform_color([0.5, 0, 0.5])  # Magenta color

        point2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point2.translate(point_goal)
        point2.paint_uniform_color([1, 0, 1])  # Magenta color

        point_center = (point_0+point_goal)/2
        point12 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        point12.translate(point_center)
        point12.paint_uniform_color([0.7, 0, 0.7])  # Magenta color

        point_obj = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point_obj.translate(self.pos_obj)
        point_obj.paint_uniform_color([0.7, 0, 0.7])  # Magenta color

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(self.point_cloud_data_wo_floor)

        # Initialize additional markers if cropped point cloud is not empty
        ins1, ins2, ins3, angles, rotation_matrix = [0.0], [-1.0], [-1.0], [0.0], -1

        side_info = []
        points_side1 = []
        points_side2 = []
        if self.point_cloud_data_wo_floor.size != 0:
            # Create an oriented bounding box and get the points inside
            angle = 0.0 #use to rotate the box around the principle axis

            #keep a global representation of the computed distances, independent of the direction of motion
            if point_0[0] <= point_goal[0]:
                obb, rotation_matrix, z_axis = self.create_obb(point_0, point_goal,angle)
            else:
                obb, rotation_matrix, z_axis = self.create_obb(point_goal, point_0,angle)
            #self.get_logger().info(f"rotation_matrix: {rotation_matrix}, z axis: {z_axis} sign: {np.sign(z_axis)}")
            # Rotate cropped points for calculations
            point_0_rotated = point_0 @ rotation_matrix.T
            point_goal_rotated = point_goal @ rotation_matrix.T
            point_center_rotated = point_center @ rotation_matrix.T
            dist1 = abs(point_goal_rotated[0] - point_0_rotated[0])
            point_cloud_rotated = self.point_cloud_data_wo_floor @ rotation_matrix.T

            self.get_logger().info(f"rotated point 0: {point_0_rotated}, point goal: {point_goal_rotated} dist1: {dist1}")

            min_p_id_x = np.argmin(point_cloud_rotated[:, 0])
            max_p_id_x = np.argmax(point_cloud_rotated[:, 0])
            min_p_id_y = np.argmin(point_cloud_rotated[:, 1])
            max_p_id_y = np.argmax(point_cloud_rotated[:, 1])
            max_p_id_z = np.argmax(point_cloud_rotated[:, 2])
            min_p_id_z = np.argmin(point_cloud_rotated[:, 2])

            # Define bounding points based on the rotated cloud
            min_p_x = self.point_cloud_data_wo_floor[min_p_id_x]
            max_p_x = self.point_cloud_data_wo_floor[max_p_id_x]
            min_p_y = self.point_cloud_data_wo_floor[min_p_id_y]
            max_p_y = self.point_cloud_data_wo_floor[max_p_id_y]
            max_p_z = self.point_cloud_data_wo_floor[max_p_id_z]
            min_p_z = self.point_cloud_data_wo_floor[min_p_id_z]
            min_max_xyz=np.array([min_p_x,max_p_x,min_p_y,max_p_y,min_p_z,max_p_z])

            min_p_x_rotated = np.min(point_cloud_rotated[:, 0])
            max_p_x_rotated = np.max(point_cloud_rotated[:, 0])
            min_p_y_rotated = np.min(point_cloud_rotated[:, 1])
            max_p_y_rotated = np.max(point_cloud_rotated[:, 1])
            min_p_z_rotated = np.min(point_cloud_rotated[:, 2])
            max_p_z_rotated = np.max(point_cloud_rotated[:, 2])
            self.get_logger().info(f"min y: {min_p_y_rotated}, max y: {max_p_y_rotated} min z: {min_p_z_rotated}")
            # Create small spheres at key points
            point3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            point3.translate(min_p_x)
            point4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            point4.translate(max_p_x)
            point5 = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            point5.translate(min_p_y)
            point6 = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            point6.translate(max_p_y)
            point7 = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            point7.translate(min_p_z)

            # Color the points for visualization
            point3.paint_uniform_color([0.5, 0, 0])  # Red
            point4.paint_uniform_color([1, 0, 0])  # Red
            point5.paint_uniform_color([0.1, 0.4, 0.7])  #
            point6.paint_uniform_color([0.2, 0.7, 1.0])  #
            point7.paint_uniform_color([0, 0, 1])  # blue

            # Calculate in1, in2, in3 based on rotated points
             #min_z because rotated bounding box is initialized upside down
            in1_1 = -(point_0_rotated[1] - max_p_y_rotated) / dist1
            in1_2 = (point_0_rotated[1] - min_p_y_rotated) / dist1
            in1_3 = (point_0_rotated[2] - min_p_z_rotated) / dist1
            in2 = abs(point_0_rotated[0] - min_p_x_rotated) / dist1
            in3 = abs(point_0_rotated[0] - max_p_x_rotated) / dist1
            ins1 = [in1_1,in1_2,in1_3]
            ins2 = (np.ones(3)*in2).tolist()
            ins3 = (np.ones(3)*in3).tolist()
            self.get_logger().info(f'ins1: {ins1}, ins2: {ins2}, ins3: {ins3}')
            angles = [0.0,0.0,0.0]
            # Combine all elements for visualization

            all_point_clouds = [obb, point1, point2, point3, point4, point5, point6, point7, self.obj_bb] + self.obst_bbs #, point6, point7

        else:
            # If no additional points are detected in the box, only visualize the main points
            self.get_logger().info(f'Empty crop')
            #self.get_logger().info(f'point_0: {point_0}, point_goal: {point_goal}')
            all_point_clouds = [point12, point1, point2]
            #o3d.visualization.draw_geometries(all_point_clouds)

        if self.get_parameter('visualize_obst_view').value:
            self.get_logger().info(f"obstacle bb: {self.obst_bbs}")
            o3d.visualization.draw_geometries([self.o3d_cloud_obst_obj,*all_point_clouds]) #,*self.joint_points
        elif self.get_parameter('visualize_obst_view_all').value:
                self.get_logger().info(f"obstacle bb: {self.obst_bbs}")
                o3d.visualization.draw_geometries([self.o3d_cloud_all,*all_point_clouds]) #,*self.joint_points
        return ins1, ins2, ins3, angles,rotation_matrix,min_max_xyz


    def transform_point_cloud(self,points,camera_pos,camera_rot):
        roll, pitch, yaw = camera_rot[0], camera_rot[1], camera_rot[2]

        # Compute rotation matrix
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = R_yaw @ R_pitch @ R_roll  # Combine rotations

        # Transformation matrix
        T_camera_to_world = np.eye(4)
        T_camera_to_world[:3, :3] = R
        T_camera_to_world[:3, 3] = camera_pos

        ones = np.ones((points.shape[0], 1))  # Homogeneous coordinates
        points_homogeneous = np.hstack((points, ones))  # Add 1s
        points_transformed = (T_camera_to_world @ points_homogeneous.T).T  # Transform

        return points_transformed[:, :3]  # Remove homogeneous coordinate

    def remove_robot_links_from_point_cloud(self, point_cloud, joint_poses, granularity=10):
        """
        Remove points near the connecting links (capsules) between joints, with an offset for the link between joint 0 and 1.

        :param point_cloud: Numpy array of shape (N, 3) representing the point cloud.
        :param joint_poses: List of poses for each joint (geometry_msgs/Pose).
        :param granularity: Number of intermediate points along each link for finer checking.
        :return: Filtered point cloud (Numpy array of shape (M, 3), where M <= N).
        """
        filtered_points = []
        joint_positions = []
        joint_spheres = []

        # Extract joint positions
        for joint_pose in joint_poses:
            joint_position = np.array([-joint_pose.position.x, -joint_pose.position.y, joint_pose.position.z])
            joint_positions.append(joint_position)

            # Visualize joints as small spheres for debugging
            joint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            joint_sphere.translate(joint_position)
            joint_spheres.append(joint_sphere)

        # Convert to numpy array for vectorized operations
        joint_positions = np.array(joint_positions)

        # Precompute link interpolation points
        link_points = []
        for i in range(len(joint_positions) - 1):
            start = joint_positions[i]
            end = joint_positions[i + 1]

            if i == 0:
                # Offset for the link between joint 0 and 1
                direction = end - start
                direction_unit = direction / np.linalg.norm(direction)

                # Project direction onto the X-Y plane
                direction_xy = direction.copy()
                direction_xy[2] = 0  # Zero out the Z component
                direction_unit_xy = direction_xy / np.linalg.norm(direction_xy)

                # Convert quaternion to rotation matrix
                quaternion = [self.joint0_quat.x, self.joint0_quat.y, self.joint0_quat.z, self.joint0_quat.w]
                rotation = R.from_quat(quaternion)

                # Define the base offset vector in the X-Y plane
                base_offset = np.array([self.link0_offset, 0, 0])  # Offset along X-axis
                # Extract the Z-axis (X-Y plane) rotation from the quaternion
                rotation_xy = R.from_euler('z', R.from_quat(quaternion).as_euler('xyz')[2])  # Only Z-axis rotation
                rotated_offset_xy = rotation_xy.apply(base_offset)

                # Apply additional rotation for link0_offset_angle
                angle_rotation = R.from_euler('z', self.link0_offset_angle)  # Rotation by link0_offset_angle
                final_offset_xy = angle_rotation.apply(rotated_offset_xy)

                # Apply the X-Y offset to start and end points
                control_point_1 = start + np.array([final_offset_xy[0], final_offset_xy[1], 0])
                control_point_2 = end + np.array([final_offset_xy[0], final_offset_xy[1], 0])

                # Interpolate along the path (start → control_point_1 → control_point_2 → end)
                for t in np.linspace(0, 1, granularity):
                    if t <= 0.33:
                        interp_point = (1 - 3 * t) * start + 3 * t * control_point_1
                    elif t <= 0.66:
                        interp_point = (1 - 3 * (t - 0.33)) * control_point_1 + 3 * (t - 0.33) * control_point_2
                    else:
                        interp_point = (1 - 3 * (t - 0.66)) * control_point_2 + 3 * (t - 0.66) * end
                    link_points.append(interp_point)
            else:
                # Linear interpolation for other links
                for t in np.linspace(0, 1, granularity):
                    interp_point = (1 - t) * start + t * end
                    link_points.append(interp_point)

        link_points = np.array(link_points)
        link_radius = self.robot_link_radius

        # Vectorized point-to-link distance computation
        distances = np.linalg.norm(point_cloud[:, None, :] - link_points[None, :, :], axis=2)
        min_distances = np.min(distances, axis=1)

        # Filter points outside the radius
        mask = min_distances > link_radius
        filtered_points = point_cloud[mask]

        for link_point in link_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(link_point)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Red for link points
            joint_spheres.append(sphere)

        return filtered_points, joint_spheres

    def point_to_line_segment_distance(self, point, start, end):
        """
        Compute the shortest distance from a point to a line segment.

        :param point: Numpy array of shape (3,) representing the point.
        :param start: Numpy array of shape (3,) representing the start of the segment.
        :param end: Numpy array of shape (3,) representing the end of the segment.
        :return: Shortest distance from the point to the line segment.
        """
        line_vec = end - start
        point_vec = point - start

        # Project point onto the line (parameterized by t)
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            # Start and end are the same point
            return np.linalg.norm(point - start)

        t = np.dot(point_vec, line_vec) / line_len
        t = np.clip(t, 0, 1)  # Ensure t is within [0, 1]

        # Find the closest point on the line segment
        closest_point = start + t * line_vec

        # Compute distance to the closest point
        return np.linalg.norm(point - closest_point)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
