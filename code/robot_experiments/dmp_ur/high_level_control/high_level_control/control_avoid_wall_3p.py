import os
import math
import copy
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from custom_interfaces.srv import ReachPoint, CheckTrajectory
from custom_interfaces.srv import ExecuteTrajectory, ExecuteDmp, ExecuteDmpAuto, GetAvoidanceParams, ExecuteRrtTrajectory, GetBoundingBoxParams
from ur_fk.msg import JointPoses
import numpy as np

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion

from gazebo_ros_link_attacher.srv import Attach

from std_srvs.srv import Empty
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

from dmp import dmp_cartesian as dmp

import torch
import torch.nn as nn

from ament_index_python.packages import get_package_share_directory

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to Euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Below should be replaced when porting for ROS2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    q = np.array([x,y,z,w])
    magnitude = np.linalg.norm(q)
    # Normalize each component by dividing by the magnitude
    normalized_q = q / magnitude
    x,y,z,w = normalized_q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return Quaternion(w=w, x=x, y=y, z=z)

class HighLevelControlNode(Node):
    def __init__(self):
        super().__init__('high_level_control_node')
        self.pending_futures = []
        self.tol = 0.005 #0.001
        self.planner = None
        self.declare_parameter('drop_height_offset',0.00)
        self.gripper_width = 0.16
        self.gripper_length = 0.18 #from wrist3 ur joint
        self.gripper_depth = 0.03 #
        self.grasped_object_offset = 0.00
        self.uncertainty_offset = 0.00
        #self.x_offset = -0.027
        self.pose_eef_init = Pose()
        self.pose_eef_init.position.x = 0.5 #0.4 #+ self.x_offset
        self.pose_eef_init.position.y = -0.5 #-0.5 #-0.39#,-0.49
        self.pose_eef_init.position.z = 0.18 #175 #12#0.08#,0.112 #corresponds to the pick height of a 5cm tall cube
        self.eef_roll = 0.0#0.8
        self.eef_pitch = np.pi
        self.eef_yaw = np.pi/2
        self.pose_eef_init.orientation = euler_to_quaternion(self.eef_roll,self.eef_pitch,self.eef_yaw)#,euler_to_quaternion(0,np.pi,0)
        self.drop_height = self.pose_eef_init.position.z + 0.15

        pkg_path = get_package_share_directory("high_level_control")

        print("high_level_control package path: ",pkg_path)
        #file_path = os.path.join(pkg_path, "docs", "dmp", "straight_slow_1D.txt")
        #self.git_dir = get_package_share_directory("dmp_ur")
        self.exp_dir2 = os.path.join(pkg_path, "docs", "experiments", "pickndrop_3p")
        self.exp_dir1 = os.path.join(pkg_path, "docs", "experiments", "avoidance_3p")
        self.dmp_docs = os.path.join(pkg_path, "docs", "dmp")
        self.traj_file_path = os.path.join(pkg_path, "docs", "dmp", "straight_slow_1D.txt")
        self.get_logger().info("Creating the DMP...")
        self.dmp = self.initialize_dmp(self.traj_file_path,n_dmps=3, n_bfs=10,K=25,rescale='rotodilatation', basis='rbf', alpha_s=4.0, tol=0.001)

        self.learned_L=0.15 #np.linalg.norm(self.dmp.learned_position) #
        self.tau = 0.5

        max_in = 1.0
        self.ins_offsets = np.array([0.02,-0.02,0.02]) #evaluated in tests that these offsets ensure successful avoidance in 99.5% of cases for this nn model
        self.nn_model_name = f"/model_50_3p_{max_in}.pth"
        self.get_logger().info(f"DMP type: {type(self.dmp)}, Loading the NN...")
        self.model = torch.jit.load(self.dmp_docs + self.nn_model_name)
        self.model.eval()

        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_detection_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv_cb_group = MutuallyExclusiveCallbackGroup()

        self.cli_reach_point = self.create_client(ReachPoint, '/low_level_control/reach_point',callback_group=self.client_cb_group)
        self.cli_execute_traj = self.create_client(ExecuteTrajectory, '/low_level_control/execute_trajectory',callback_group=self.client_cb_group)
        self.cli_execute_rrt_traj = self.create_client(ExecuteRrtTrajectory, '/ur5e_moveit_commander/execute_rrt_trajectory',callback_group=self.client_cb_group)
        self.cli_check_traj = self.create_client(CheckTrajectory, '/ur5e_moveit_commander/check_trajectory',callback_group=self.client_cb_group)
        self.cli_get_avoidance_params = self.create_client(GetAvoidanceParams, '/image_processing/get_avoidance_params',callback_group=self.client_detection_cb_group)
        self.cli_get_bounding_box_params = self.create_client(GetBoundingBoxParams, '/image_processing/get_bounding_box_params',callback_group=self.client_detection_cb_group)
        self.cli_gripper_close = self.create_client(Empty, '/high_level_control/close_gripper',callback_group=self.client_cb_group)
        self.cli_gripper_open = self.create_client(Empty, '/high_level_control/open_gripper',callback_group=self.client_cb_group)
        self.start_up_service = self.create_service(Empty, '/high_level_control/go_to_start', self.start_up_callback,callback_group=self.srv_cb_group)
        self.execute_dmp_traj_service = self.create_service(ExecuteDmp, '/high_level_control/execute_dmp', self.execute_dmp_callback,callback_group=self.srv_cb_group)
        self.execute_dmp_traj_auto_service = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_dmp_automated', self.execute_dmp_automated_callback,callback_group=self.srv_cb_group)
        self.execute_rrt_traj_auto_service = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_rrt_automated', self.execute_rrt_automated_callback,callback_group=self.srv_cb_group)
        self.execute_dmp_traj_auto_service_ = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_pick_and_drop', self.execute_pick_and_drop_callback,callback_group=self.srv_cb_group)
        self.execute_linear_traj_auto_service_ = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_pick_and_drop_linear', self.execute_pick_and_drop_linear_callback,callback_group=self.srv_cb_group)
        self.execute_rrt_traj_auto_service_ = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_pick_and_drop_rrt', self.execute_pick_and_drop_rrt_callback,callback_group=self.srv_cb_group)
        self.execute_stack_traj_auto_service_ = self.create_service(ExecuteDmpAuto, '/high_level_control/execute_pick_and_stack', self.execute_pick_and_stack_callback,callback_group=self.srv_cb_group)

        self.attach_client = self.create_client(Attach, '/attach',callback_group=self.client_cb_group)
        self.detach_client = self.create_client(Attach, '/detach',callback_group=self.client_cb_group)

        self.eef_pose = None
        self.joint_poses = None
        self.joint_states = None  # Initialize the joint_states variable
        self.joint_vels = None
        self.traj_counter = 0
        self.save_joint_poses = False
        self.create_subscription(JointPoses, '/ur5e/all_joint_poses', self.eef_pose_callback, 10,callback_group=self.sub_cb_group)
        self.create_subscription(Bool, '/low_level_control/trajectory_complete', self.trajectory_complete_callback, 10,callback_group=self.sub_cb_group)
        self.create_subscription(JointState,'/joint_states',self.joint_state_callback,10,callback_group=self.sub_cb_group)
        self.get_logger().info("high level control node started.")

        self.rrt_planning_time = None
        self.rrt_planning_attempts = None
        self.rrt_success = False
        self.rrt_q_traj = None
        self.rrt_dq_traj = None
        self.rrt_ddq_traj = None

        self.trajectory_completed = None

    def eef_pose_callback(self,msg):
        self.joint_poses = msg.poses
        self.eef_pose = msg.poses[-1] #get the wrist3 (eef) pose

    def joint_state_callback(self,msg):
        self.joint_states = msg.position
        self.joint_vels = msg.velocity

    def start_up_callback(self,req,res):
        if not self.cli_reach_point.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /reach_point not available!')
            return res

        request = ReachPoint.Request()
        request.x = self.pose_eef_init.position.x
        request.y = self.pose_eef_init.position.y
        request.z = self.pose_eef_init.position.z
        request.qr = self.eef_roll
        request.qp = self.eef_pitch
        request.qy = self.eef_yaw

        result = self.cli_reach_point.call(request)
        #future = self.cli_reach_point.call_async(request)

        # Wait for the result
        #rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        #if future.result() is not None:
        #    result = future.result()
        self.get_logger().info(f'Start position is requested: {result.success}')
        #else:
        #    self.get_logger().error('Service call failed or timed out.')

        return res

    def execute_dmp_callback(self,req,res):
        #self.get_logger().info("entering dmp traj service...")
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        gripper_pose = self.eef_pose.deepcopy()
        gripper_pose.position.z -= self.gripper_length
        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        goal_position = initial_position + np.array([-0.1, 0.0, 0.0])
        in1 = req.in1

        if req.goal_x != 0.0:
            goal_position[0] = req.goal_x
        if req.goal_y != 0.0:
            goal_position[1] = req.goal_y
        if req.goal_z != 0.0:
            goal_position[1] = req.goal_z

        traj_angle = req.angle

        self.dmp.x_0 = initial_position
        self.dmp.x_goal = goal_position
        self.dmp, _ = self.get_new_weights(self.dmp,self.model,[in1],traj_angle)
        #self.tau = self.perturb_tau(dist,[in1])
        x_traj,_,_,_,_,_ = self.dmp.rollout(tau=self.tau)
        pose_traj = self.create_pose_trajectory(x_traj,self.eef_pose.orientation)
        request = ExecuteTrajectory.Request()
        request.trajectory = pose_traj
        result = self.cli_execute_traj.call(request)

        self.get_logger().info(f'Cartesian DMP trajectory execution is requested: {result.success}')

        return res

    def execute_dmp_automated_callback(self,req,res):
        #self.get_logger().info("entering automated dmp traj service...")
        computation_times = {}
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        start_time_exec = time.time()
        start_time = int(start_time_exec)
        current_dir = f'{self.exp_dir1}/{start_time}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length
        req_detection = GetAvoidanceParams.Request()
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose

        goal_position = initial_position.copy()
        goal_position[0] = -initial_position[0]
        #goal_position[2] += 0.15
        if req.goal_x != 0.0:
            goal_position[0] = -req.goal_x #account for robot rotated 180 degrees around Z compared to world frame
        if req.goal_y != 0.0:
            goal_position[1] = -req.goal_y
        if req.goal_z != 0.0:
            goal_position[2] = req.goal_z + self.gripper_length

        goal_pose = copy.deepcopy(gripper_pose)
        goal_pose.position.x = goal_position[0]
        goal_pose.position.y = goal_position[1]
        goal_pose.position.z = goal_position[2] - self.gripper_length
        req_detection.pose_goal = goal_pose
        req_detection.obj_dims = req.obj_dims

        result_detection = self.cli_get_avoidance_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        if goal_pose.position.x != 0.0:
            goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z])
        ins1 = result_detection.ins1
        ins2 = result_detection.ins2
        ins3 = result_detection.ins3
        dist = np.linalg.norm(goal_position-initial_position)
        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        angles = result_detection.angles



        if sol_id >= len(ins1):
            self.get_logger().error(f'Requested solution id is larger than the provided number solutions! Abort...')
            res.success = False
            return res

        ins1[0]+=self.gripper_width/2/dist+self.uncertainty_offset
        ins1[1]+=self.gripper_width/2/dist
        if sol_id == -1:
            sol_id = np.argmin(ins1)

        if sol_id == 0:
            self.get_logger().info(f"Attemping to avoid obstacle around to the front")
        elif sol_id == 1:
            self.get_logger().info(f"Attemping to avoid obstacle around behind")
        elif sol_id == 2:
            self.get_logger().info(f"Attemping to avoid over the obstacle")

        self.get_logger().info(f"Available ins1: {ins1}, ins2: {ins2}, ins3: {ins3}, sol_id: {sol_id}")
        in1 = ins1[sol_id]
        #assign the smaller value to in2 and the larger value to in3
        if ins2[sol_id] > ins3[sol_id]:
            in2 = ins3[sol_id]-self.gripper_depth/2/dist
            in3 = ins2[sol_id]+self.gripper_depth/2/dist
        else:
            in2 = ins2[sol_id]-self.gripper_depth/2/dist
            in3 = ins3[sol_id]+self.gripper_depth/2/dist

        traj_angle = 0.0
        angle_swap = 1.0 #same angle goes always in the same direction from the start point perspective, which is different to the same perspective of the detection
        if goal_position[0] > initial_position[0]:
            angle_swap = -1.0

        if sol_id == 0:
            traj_angle = angle_swap*np.pi/2
        elif sol_id == 1:
            traj_angle = -angle_swap*np.pi/2
        #elif req.sol_id == 2:
        #    in1+=self.gripper_length

        if in1 < 0.0:
            in1 = 0.0
        ins = np.array([in1,in2,in3]) + self.ins_offsets
        #in1 = in1/dist
        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        self.dmp.x_0 = initial_position
        self.dmp.x_goal = goal_position
        start_time = time.time()
        self.dmp, _ = self.get_new_weights(self.dmp,self.model,ins,traj_angle)
        computation_times['nn_inference'] = time.time() - start_time
        #self.tau = self.perturb_tau(dist,[in1])
        start_time = time.time()
        x_traj,dx_traj,ddx_traj,_,_,_ = self.dmp.rollout(tau=self.tau)
        computation_times['dmp_rollout'] = time.time() - start_time
        start_time = time.time()
        pose_traj = self.create_pose_trajectory(x_traj,self.eef_pose.orientation)
        computation_times['gen_pose_traj'] = time.time() - start_time
        #self.trajectory_completed = False
        request_traj = ExecuteTrajectory.Request()
        request_traj.trajectory = pose_traj
        result_traj = self.cli_execute_traj.call(request_traj)
        q_traj = result_traj.joint_trajectory
        self.get_logger().info(f'Cartesian DMP trajectory execution is requested with angle {traj_angle} and ins: {ins}: {result_traj.success}')
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        s = current_dir + 'traj_data.npz'
        np.savez(s, computation_times=computation_times, time_stamps=time_stamps,q_traj=q_traj,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj, x_traj=x_traj,dx_traj=dx_traj,ddx_traj=ddx_traj,ins=ins,pos_init=initial_position,pos_goal=goal_position,angle_swap=angle_swap,sol_id=sol_id,sol_id_req=req.sol_id,traj_angle=traj_angle,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)
        return res

    def execute_rrt_automated_callback(self,req,res):
        #self.get_logger().info("entering automated dmp traj service...")
        computation_times = {}
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        start_time_exec = time.time()
        start_time = int(start_time_exec)
        current_dir = f'{self.exp_dir1}_rrt/{start_time}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length

        goal_position = initial_position.copy()
        goal_position[0] = -initial_position[0]
        #goal_position[2] += 0.15
        if req.goal_x != 0.0:
            goal_position[0] = -req.goal_x #account for robot rotated 180 degrees around Z compared to world frame
        if req.goal_y != 0.0:
            goal_position[1] = -req.goal_y
        if req.goal_z != 0.0:
            goal_position[2] = req.goal_z + self.gripper_length

        goal_pose = copy.deepcopy(gripper_pose)
        goal_pose.position.x = goal_position[0]
        goal_pose.position.y = goal_position[1]
        goal_pose.position.z = goal_position[2] - self.gripper_length


        req_detection = GetBoundingBoxParams.Request()
        req_detection.pose_goal = goal_pose
        req_detection.obj_dims = req.obj_dims
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose
        self.get_logger().info("calling get bounding box...")
        result_detection = self.cli_get_bounding_box_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        if goal_pose.position.x != 0.0:
            goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z+self.get_parameter('drop_height_offset').value])
        #goal_pose.position.x = goal_position[0]
        #goal_pose.position.y = goal_position[1]
        goal_pose.position.z = goal_position[2]
        goal_pose.orientation = self.pose_eef_init.orientation
        obstacle_poses = result_detection.obstacle_poses
        obstacle_sizes = result_detection.obstacle_sizes
        self.get_logger().info(f"received bounding box params, length of obstacles: {len(obstacle_sizes)}")
        dist = np.linalg.norm(goal_position-initial_position)
        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')

        #self.trajectory_completed = False
        request_traj = ExecuteRrtTrajectory.Request()
        request_traj.target_pose = goal_pose
        request_traj.obstacle_poses = obstacle_poses
        request_traj.obstacle_sizes = obstacle_sizes
        self.get_logger().info(f'Calling cli_execute_rrt_traj...')
        #result_traj = self.cli_execute_rrt_traj.call(request_traj)
        # Call the client asynchronously
        future = self.cli_execute_rrt_traj.call_async(request_traj)

        # Store the future so it’s not garbage-collected
        self.pending_futures.append(future)

        # Add a callback for when the service responds
        future.add_done_callback(self.handle_client_response)

        self.get_logger().info(f'rrt trajectory execution is requested')
        start_time = time.time()
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        computation_times['traj_execution'] = time.time() - start_time

        s = current_dir + 'traj_data.npz'
        np.savez(s, computation_times=computation_times, time_stamps=time_stamps,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj,
        pos_init=initial_position,pos_goal=goal_position,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)
        return res


    def execute_pick_and_drop_callback(self,req,res): #pick and drop variation
        #self.get_logger().info("entering automated dmp traj service...")
        #start_time00 = time.time()
        start_time_exec = time.time()
        self.planner = "dmp"
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        total_time = 0.0
        total_computation_time = 0.0
        computation_times = {}

        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length

        start_time_int = int(start_time_exec)
        current_dir = f'{self.exp_dir2}/{start_time_int}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        obj_dims = req.obj_dims
        req_detection = GetAvoidanceParams.Request()
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose
        req_detection.obj_dims = obj_dims
        req_detection.time_stamp = start_time
        self.get_logger().info(f"start_time: {time.time()-start_time}")
        result_detection = self.cli_get_avoidance_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z+self.get_parameter('drop_height_offset').value])
        ins1 = result_detection.ins1
        ins2 = result_detection.ins2
        ins3 = result_detection.ins3
        obstacle_poses = result_detection.obstacle_poses
        obstacle_sizes = result_detection.obstacle_sizes
        detection_time_stamp_start = result_detection.time_stamp_start
        detection_time_stamp_end = result_detection.time_stamp_end
        self.get_logger().info(f"detection communication time: {time.time() - detection_time_stamp_end}, measured time from control: {computation_times['get_avoidance_params']}, measured time from detection: {detection_time_stamp_end-detection_time_stamp_start}, communication to detection node: {detection_time_stamp_start-start_time}")
        dist = np.linalg.norm(goal_position-initial_position)


        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        angles = result_detection.angles

        if sol_id >= len(ins1):
            self.get_logger().error(f'Requested solution id is larger than the provided number solutions! Abort...')
            res.success = False
            return res

        attached_obj_name = 'cube'
        if req.attached_obj_name != '':
            attached_obj_name = req.attached_obj_name

        self.attach_cube(attached_obj_name)
        time.sleep(0.05)
        if not self.cli_gripper_close.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /close_gripper not available!')
            return res

        self.cli_gripper_close.call(Empty.Request())
        time.sleep(0.05)
        self.get_logger().info(f'Closing gripper')

        ins1[0]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[1]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[2]+=self.grasped_object_offset/dist #grasped object can only collide when avoiding over the obstacle, as it is small
        #ins1[2]+=np.maximum(0,obj_dims[2]-0.02)/dist #currently, the grasp is at the bottom of the object (same as bottom of gripper)
        if sol_id == -1:
            sol_id = np.argmin(ins1)

        if sol_id == 0:
            self.get_logger().info(f"Attemping to avoid obstacle around to the front")
        elif sol_id == 1:
            self.get_logger().info(f"Attemping to avoid obstacle around behind")
        elif sol_id == 2:
            self.get_logger().info(f"Attemping to avoid over the obstacle")

        self.get_logger().info(f"Available ins1: {ins1}, ins2: {ins2}, ins3: {ins3}, sol_id: {sol_id}")
        in1 = ins1[sol_id]
        in2 = ins2[sol_id]-self.gripper_depth/2/dist
        in3 = ins3[sol_id]+self.gripper_depth/2/dist

        traj_angle = 0.0
        angle_swap = 1.0 #same angle goes always in the same direction from the start point perspective, which is different to the same perspective of the detection
        if goal_position[0] > initial_position[0]:
            angle_swap = -1.0

        if sol_id == 0:
            traj_angle = angle_swap*np.pi/2
        elif sol_id == 1:
            traj_angle = -angle_swap*np.pi/2

        if in1 < 0.0:
            in1 = 0.0
        ins = np.array([in1,in2,in3]) + self.ins_offsets
        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        self.dmp.x_0 = initial_position
        self.dmp.x_goal = goal_position
        start_time = time.time()
        self.dmp, _ = self.get_new_weights(self.dmp,self.model,ins,traj_angle)
        computation_times['nn_inference'] = time.time() - start_time
        #self.tau = self.perturb_tau(dist,[in1])
        start_time = time.time()
        x_traj,dx_traj,ddx_traj,_,_,_ = self.dmp.rollout(tau=self.tau)
        computation_times['dmp_rollout'] = time.time() - start_time
        start_time = time.time()
        pose_traj = self.create_pose_trajectory_moveit(x_traj,self.eef_pose.orientation)
        computation_times['gen_pose_traj'] = time.time() - start_time

        start_time = time.time()
        request_check = CheckTrajectory.Request()
        request_check.trajectory = pose_traj
        request_check.obstacle_poses = obstacle_poses
        request_check.obstacle_sizes = obstacle_sizes
        #result_check = self.cli_check_traj.call(request_check)
        # Call the client asynchronously
        future = self.cli_check_traj.call_async(request_check)

        # Store the future so it’s not garbage-collected
        self.pending_futures.append(future)

        # Add a callback for when the service responds
        future.add_done_callback(self.handle_client_response)

        '''computation_times['collision_check'] = time.time() - start_time
        if not result_check.success:
            self.get_logger().error('Collision check was not successful, abort!')
            return res
        self.trajectory_completed = False
        request_traj = ExecuteTrajectory.Request()
        request_traj.trajectory = pose_traj
        result_traj = self.cli_execute_traj.call(request_traj)
        q_traj_multiarray = result_traj.joint_trajectory
        # Create an empty list to hold the rows
        q_traj = []
        # Iterate over the Float64MultiArray[] message
        for multi_array in q_traj_multiarray:
            # Convert the data from each Float64MultiArray into a numpy array
            q_traj.append(np.array(multi_array.data))
        # Stack the individual numpy arrays into a 2D numpy array
        if len(q_traj) > 0:
            q_traj = np.stack(q_traj)'''

        #print(q_traj.shape)
        #total_computation_time = time.time() - start_time_exec
        self.get_logger().info(f'Cartesian DMP trajectory execution is requested with angle {traj_angle} and ins: {ins}')
        start_time = time.time()
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        computation_times['traj_execution'] = time.time() - start_time
        if not self.cli_gripper_open.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /open_gripper not available!')
            return res

        self.cli_gripper_open.call(Empty.Request())
        self.get_logger().info(f'Opening gripper')

        time.sleep(0.05)
        self.detach_cube(attached_obj_name)
        total_time = time.time() - start_time_exec
        s = current_dir + 'traj_data.npz'
        #total_exec_time = time.time()-start_time00
        self.get_logger().info(f"Finished successfully! Totoal execution time: {total_time}, total processing time: {total_time-computation_times['traj_execution']}")
        np.savez(s,uncertainty_offset=self.uncertainty_offset,grasped_object_offset=self.grasped_object_offset,planner=self.planner,total_time=total_time,q_traj=self.rrt_q_traj,dq_traj=self.rrt_dq_traj,ddq_traj=self.rrt_ddq_traj,attempts=self.rrt_planning_attempts,planning_time=self.rrt_planning_time,obstacle_poses=obstacle_poses,obstacle_sizes=obstacle_sizes,computation_times=computation_times, time_stamps=time_stamps,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj, x_traj=x_traj,dx_traj=dx_traj,ddx_traj=ddx_traj,ins=ins,pos_init=initial_position,pos_goal=goal_position,angle_swap=angle_swap,sol_id=sol_id,sol_id_req=req.sol_id,traj_angle=traj_angle,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)
        return res

    def execute_pick_and_drop_linear_callback(self,req,res): #pick and drop variation
        #self.get_logger().info("entering automated dmp traj service...")
        start_time_exec = time.time()
        self.planner = "linear"
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        total_time = 0.0
        total_computation_time = 0.0
        computation_times = {}

        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length

        start_time = int(start_time_exec)
        current_dir = f'{self.exp_dir2}_linear/{start_time}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        obj_dims = req.obj_dims
        req_detection = GetAvoidanceParams.Request()
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose
        req_detection.obj_dims = obj_dims

        result_detection = self.cli_get_avoidance_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z+self.get_parameter('drop_height_offset').value])
        ins1 = result_detection.ins1
        ins2 = result_detection.ins2
        ins3 = result_detection.ins3
        dist = np.linalg.norm(goal_position-initial_position)
        obstacle_poses = result_detection.obstacle_poses
        obstacle_sizes = result_detection.obstacle_sizes
        detection_time_stamp_end = result_detection.time_stamp_end
        self.get_logger().info(f'detection communication time: {time.time() - detection_time_stamp_end}')

        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        angles = result_detection.angles

        if sol_id >= len(ins1):
            self.get_logger().error(f'Requested solution id is larger than the provided number solutions! Abort...')
            res.success = False
            return res

        attached_obj_name = 'cube'
        if req.attached_obj_name != '':
            attached_obj_name = req.attached_obj_name

        self.attach_cube(attached_obj_name)
        time.sleep(0.05)
        if not self.cli_gripper_close.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /close_gripper not available!')
            return res

        self.cli_gripper_close.call(Empty.Request())
        time.sleep(0.05)
        self.get_logger().info(f'Closing gripper')

        ins1[0]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[1]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[2]+=self.grasped_object_offset/dist #grasped object can only collide when avoiding over the obstacle, as it is small
        #ins1[2]+=np.maximum(0,obj_dims[2]-0.02)/dist #currently, the grasp is at the bottom of the object (same as bottom of gripper)
        if sol_id == -1:
            sol_id = np.argmin(ins1)

        if sol_id == 0:
            self.get_logger().info(f"Attemping to avoid obstacle around to the front")
        elif sol_id == 1:
            self.get_logger().info(f"Attemping to avoid obstacle around behind")
        elif sol_id == 2:
            self.get_logger().info(f"Attemping to avoid over the obstacle")

        self.get_logger().info(f"Available ins1: {ins1}, ins2: {ins2}, ins3: {ins3}, sol_id: {sol_id}")
        in1 = ins1[sol_id]
        in2 = ins2[sol_id]-self.gripper_depth/2/dist
        in3 = ins3[sol_id]+self.gripper_depth/2/dist

        traj_angle = 0.0
        angle_swap = 1.0 #same angle goes always in the same direction from the start point perspective, which is different to the same perspective of the detection
        if goal_position[0] > initial_position[0]:
            angle_swap = -1.0

        if sol_id == 0:
            traj_angle = angle_swap*np.pi/2
        elif sol_id == 1:
            traj_angle = -angle_swap*np.pi/2

        if in1 < 0.0:
            in1 = 0.0
        ins = np.array([in1,in2,in3]) + self.ins_offsets
        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')

        start_time = time.time()
        x_traj = self.generate_three_segment_trajectory(initial_position, goal_position, ins[0], ins[1], ins[2], sol_id)
        computation_times['linear_traj_generation'] = time.time() - start_time
        start_time = time.time()
        pose_traj = self.create_pose_trajectory_moveit(x_traj,self.eef_pose.orientation)
        computation_times['gen_pose_traj'] = time.time() - start_time
        #self.trajectory_completed = False

        request_check = CheckTrajectory.Request()
        request_check.trajectory = pose_traj
        request_check.obstacle_poses = obstacle_poses
        request_check.obstacle_sizes = obstacle_sizes
        # Call the client asynchronously
        future = self.cli_check_traj.call_async(request_check)

        # Store the future so it’s not garbage-collected
        self.pending_futures.append(future)

        # Add a callback for when the service responds
        future.add_done_callback(self.handle_client_response)
        '''request_traj = ExecuteTrajectory.Request()
        request_traj.trajectory = pose_traj
        result_traj = self.cli_execute_traj.call(request_traj)
        q_traj_multiarray = result_traj.joint_trajectory
        # Create an empty list to hold the rows
        q_traj = []
        # Iterate over the Float64MultiArray[] message
        for multi_array in q_traj_multiarray:
            # Convert the data from each Float64MultiArray into a numpy array
            q_traj.append(np.array(multi_array.data))
        # Stack the individual numpy arrays into a 2D numpy array
        if len(q_traj) > 0:
            q_traj = np.stack(q_traj)'''
        #print(q_traj.shape)
        #total_computation_time = time.time() - start_time_exec
        self.get_logger().info(f'Three-segment linear trajectory execution is requested with angle {traj_angle} and ins: {ins}')
        start_time = time.time()
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        computation_times['traj_execution'] = time.time() - start_time
        if not self.cli_gripper_open.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /open_gripper not available!')
            return res

        self.cli_gripper_open.call(Empty.Request())
        self.get_logger().info(f'Opening gripper')

        time.sleep(0.05)
        self.detach_cube(attached_obj_name)
        total_time = time.time() - start_time_exec
        s = current_dir + 'traj_data.npz'
        np.savez(s,uncertainty_offset=self.uncertainty_offset,grasped_object_offset=self.grasped_object_offset,planner=self.planner,total_time=total_time,q_traj=self.rrt_q_traj,dq_traj=self.rrt_dq_traj,ddq_traj=self.rrt_ddq_traj,attempts=self.rrt_planning_attempts,planning_time=self.rrt_planning_time,obstacle_poses=obstacle_poses,obstacle_sizes=obstacle_sizes,computation_times=computation_times, time_stamps=time_stamps,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj, x_traj=x_traj,ins=ins,pos_init=initial_position,pos_goal=goal_position,angle_swap=angle_swap,sol_id=sol_id,sol_id_req=req.sol_id,traj_angle=traj_angle,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)
        #np.savez(s,planner=self.planner,total_time=total_time,total_computation_time=total_computation_time,computation_times=computation_times, time_stamps=time_stamps,q_traj=q_traj,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj, x_traj=x_traj,ins=ins,pos_init=initial_position,pos_goal=goal_position,angle_swap=angle_swap,sol_id=sol_id,sol_id_req=req.sol_id,traj_angle=traj_angle,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)

        return res

    def execute_pick_and_drop_rrt_callback(self,req,res): #pick and drop variation
        start_time_exec = time.time()
        self.planner = "rrt_connect"
        self.get_logger().info("entering automated rrt traj service...")
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        total_time = 0.0
        total_computation_time = 0.0
        computation_times = {}

        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length

        start_time = int(start_time_exec)
        current_dir = f'{self.exp_dir2}_rrt/{start_time}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        obj_dims = req.obj_dims
        req_detection = GetBoundingBoxParams.Request()
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose
        req_detection.obj_dims = obj_dims
        self.get_logger().info("calling get bounding box...")
        result_detection = self.cli_get_bounding_box_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z+self.get_parameter('drop_height_offset').value])
        #goal_pose.position.x = goal_position[0]
        #goal_pose.position.y = goal_position[1]
        goal_pose.position.z = goal_position[2]
        goal_pose.orientation = self.pose_eef_init.orientation
        goal_pose.orientation.x = -self.pose_eef_init.orientation.x #for some reason that creates a constant orientation with moveit
        obstacle_poses = result_detection.obstacle_poses
        obstacle_sizes = result_detection.obstacle_sizes
        self.get_logger().info(f"received bounding box params, length of obstacles: {len(obstacle_sizes)}")
        dist = np.linalg.norm(goal_position-initial_position)
        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        attached_obj_name = 'cube'
        if req.attached_obj_name != '':
            attached_obj_name = req.attached_obj_name
        self.get_logger().info("attaching cube...")
        self.attach_cube(attached_obj_name)
        time.sleep(0.05)
        if not self.cli_gripper_close.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /close_gripper not available!')
            return res

        self.cli_gripper_close.call(Empty.Request())
        time.sleep(0.05)
        self.get_logger().info(f'Closing gripper')


        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')


        request_traj = ExecuteRrtTrajectory.Request()
        request_traj.target_pose = goal_pose
        request_traj.obstacle_poses = obstacle_poses
        request_traj.obstacle_sizes = obstacle_sizes
        self.get_logger().info(f'Calling cli_execute_rrt_traj...')
        #result_traj = self.cli_execute_rrt_traj.call(request_traj)
        # Call the client asynchronously
        future = self.cli_execute_rrt_traj.call_async(request_traj)

        # Store the future so it’s not garbage-collected
        self.pending_futures.append(future)

        # Add a callback for when the service responds
        future.add_done_callback(self.handle_client_response)
        total_computation_time = time.time() - start_time_exec
        self.get_logger().info(f'rrt trajectory execution is requested')
        start_time = time.time()
        while self.trajectory_completed is None: #wait for plan to be generated before waiting for the execution to finish
            time.sleep(0.001)
            #self.get_logger().info(f'waiting for moveit planning success')
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        self.rrt_success = False
        computation_times['traj_execution'] = time.time() - start_time
        if not self.cli_gripper_open.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /open_gripper not available!')
            return res

        self.cli_gripper_open.call(Empty.Request())
        self.get_logger().info(f'Opening gripper')

        time.sleep(0.05)
        self.detach_cube(attached_obj_name)
        total_time = time.time() - start_time_exec
        s = current_dir + 'traj_data.npz'
        np.savez(s,planner=self.planner,q_traj=self.rrt_q_traj,dq_traj=self.rrt_dq_traj,ddq_traj=self.rrt_ddq_traj,attempts=self.rrt_planning_attempts,planning_time=self.rrt_planning_time,obstacle_poses=obstacle_poses,obstacle_sizes=obstacle_sizes,computation_times=computation_times, time_stamps=time_stamps,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj,
        pos_init=initial_position,pos_goal=goal_position,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)

        return res

    def handle_client_response(self, future):
        try:
            result_traj = future.result()
            self.get_logger().info(f"Received result: {result_traj.success}")
            # Do something with result_traj here
            self.rrt_planning_time = result_traj.planning_time
            self.rrt_planning_attempts = result_traj.attempts
            rrt_joint_traj = result_traj.rrt_joint_traj
            self.rrt_success = result_traj.success
            self.planner = result_traj.planner
            if self.rrt_success:
                positions = []
                velocities = []
                accelerations = []
                for point in rrt_joint_traj.points:
                    positions.append(point.positions)
                    velocities.append(point.velocities)
                    accelerations.append(point.accelerations)
                self.rrt_q_traj = positions
                self.rrt_dq_traj = velocities
                self.rrt_ddq_traj = accelerations

        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def execute_pick_and_stack_callback(self,req,res): #pick and drop variation
        #self.get_logger().info("entering automated dmp traj service...")
        #start_time00 = time.time()
        start_time_exec = time.time()
        self.planner = "dmp"
        if self.eef_pose is None:
            self.get_logger().error("No eef pose received yet.")
            return
        total_time = 0.0
        total_computation_time = 0.0
        computation_times = {}

        initial_position = np.array([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z])
        #
        gripper_pose = copy.deepcopy(self.eef_pose) #detect obstacle based on the gripper pose
        gripper_pose.position.z -= self.gripper_length

        start_time_int = int(start_time_exec)
        current_dir = f'{self.exp_dir2}/{start_time_int}/'
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        start_time = time.time()
        obj_dims = req.obj_dims
        req_detection = GetAvoidanceParams.Request()
        req_detection.dir_path = current_dir
        req_detection.pose_0 = gripper_pose
        req_detection.obj_dims = obj_dims
        req_detection.time_stamp = start_time
        self.get_logger().info(f"start_time: {time.time()-start_time}")
        result_detection = self.cli_get_avoidance_params.call(req_detection)
        if not result_detection.success:
            self.get_logger().info(f'Detection failed')
            return res
        computation_times['get_avoidance_params'] = time.time() - start_time
        goal_pose = result_detection.poses[0]
        goal_position = np.array([-goal_pose.position.x,-goal_pose.position.y,goal_pose.position.z+self.pose_eef_init.position.z+self.get_parameter('drop_height_offset').value])
        ins1 = result_detection.ins1
        ins2 = result_detection.ins2
        ins3 = result_detection.ins3
        obstacle_poses = result_detection.obstacle_poses
        obstacle_sizes = result_detection.obstacle_sizes
        detection_time_stamp_start = result_detection.time_stamp_start
        detection_time_stamp_end = result_detection.time_stamp_end
        self.get_logger().info(f"detection communication time: {time.time() - detection_time_stamp_end}, measured time from control: {computation_times['get_avoidance_params']}, measured time from detection: {detection_time_stamp_end-detection_time_stamp_start}, communication to detection node: {detection_time_stamp_start-start_time}")
        dist = np.linalg.norm(goal_position-initial_position)


        #self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        #self.get_logger().info(f"position init: {initial_position}, goal position: {goal_position}, ins1: {ins1}, dist: {dist}")
        sol_id = req.sol_id

        angles = result_detection.angles

        if sol_id >= len(ins1):
            self.get_logger().error(f'Requested solution id is larger than the provided number solutions! Abort...')
            res.success = False
            return res

        attached_obj_name = 'cube'
        if req.attached_obj_name != '':
            attached_obj_name = req.attached_obj_name

        self.attach_cube(attached_obj_name)
        time.sleep(0.05)
        if not self.cli_gripper_close.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /close_gripper not available!')
            return res

        self.cli_gripper_close.call(Empty.Request())
        time.sleep(0.05)
        self.get_logger().info(f'Closing gripper')

        ins1[0]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[1]+=(self.gripper_width/2)/dist #+obj_dims[1]/2
        ins1[2]+=self.grasped_object_offset/dist #grasped object can only collide when avoiding over the obstacle, as it is small
        #ins1[2]+=np.maximum(0,obj_dims[2]-0.02)/dist #currently, the grasp is at the bottom of the object (same as bottom of gripper)
        if sol_id == -1:
            sol_id = np.argmin(ins1)

        if sol_id == 0:
            self.get_logger().info(f"Attemping to avoid obstacle around to the front")
        elif sol_id == 1:
            self.get_logger().info(f"Attemping to avoid obstacle around behind")
        elif sol_id == 2:
            self.get_logger().info(f"Attemping to avoid over the obstacle")

        self.get_logger().info(f"Available ins1: {ins1}, ins2: {ins2}, ins3: {ins3}, sol_id: {sol_id}")
        in1 = ins1[sol_id]
        in2 = ins2[sol_id]-self.gripper_depth/2/dist
        in3 = ins3[sol_id]+self.gripper_depth/2/dist

        traj_angle = 0.0
        angle_swap = 1.0 #same angle goes always in the same direction from the start point perspective, which is different to the same perspective of the detection
        if goal_position[0] > initial_position[0]:
            angle_swap = -1.0

        if sol_id == 0:
            traj_angle = angle_swap*np.pi/2
        elif sol_id == 1:
            traj_angle = -angle_swap*np.pi/2

        if in1 < 0.0:
            in1 = 0.0
        ins = np.array([in1,in2,in3]) + self.ins_offsets
        #get ins from detection service
        self.get_logger().info(f'Goal position: {goal_position}, dist: {dist}')
        self.dmp.x_0 = initial_position
        self.dmp.x_goal = goal_position
        start_time = time.time()
        self.dmp, _ = self.get_new_weights(self.dmp,self.model,ins,traj_angle)
        computation_times['nn_inference'] = time.time() - start_time
        #self.tau = self.perturb_tau(dist,[in1])
        start_time = time.time()
        x_traj,dx_traj,ddx_traj,_,_,_ = self.dmp.rollout(tau=self.tau)
        computation_times['dmp_rollout'] = time.time() - start_time
        start_time = time.time()
        pose_traj = self.create_pose_trajectory(x_traj,self.eef_pose.orientation)
        computation_times['gen_pose_traj'] = time.time() - start_time

        start_time = time.time()

        self.trajectory_completed = False
        request_traj = ExecuteTrajectory.Request()
        request_traj.trajectory = pose_traj
        result_traj = self.cli_execute_traj.call(request_traj)
        q_traj_multiarray = result_traj.joint_trajectory
        # Create an empty list to hold the rows
        q_traj = []
        # Iterate over the Float64MultiArray[] message
        for multi_array in q_traj_multiarray:
            # Convert the data from each Float64MultiArray into a numpy array
            q_traj.append(np.array(multi_array.data))
        # Stack the individual numpy arrays into a 2D numpy array
        if len(q_traj) > 0:
            q_traj = np.stack(q_traj)

        #print(q_traj.shape)
        #total_computation_time = time.time() - start_time_exec
        self.get_logger().info(f'Cartesian DMP trajectory execution is requested with angle {traj_angle} and ins: {ins}')
        start_time = time.time()
        eef_traj, time_stamps, joint_traj, joint_vel_traj = self.wait_for_trajectory_completion()
        computation_times['traj_execution'] = time.time() - start_time
        if not self.cli_gripper_open.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /open_gripper not available!')
            return res

        self.cli_gripper_open.call(Empty.Request())
        self.get_logger().info(f'Opening gripper')

        time.sleep(0.05)
        self.detach_cube(attached_obj_name)
        total_time = time.time() - start_time_exec
        s = current_dir + 'traj_data.npz'
        #total_exec_time = time.time()-start_time00
        self.get_logger().info(f"Finished successfully! Totoal execution time: {total_time}, total processing time: {total_time-computation_times['traj_execution']}")
        np.savez(s,grasped_object_offset=self.grasped_object_offset,planner=self.planner,total_time=total_time,q_traj=self.rrt_q_traj,dq_traj=self.rrt_dq_traj,ddq_traj=self.rrt_ddq_traj,attempts=self.rrt_planning_attempts,planning_time=self.rrt_planning_time,obstacle_poses=obstacle_poses,obstacle_sizes=obstacle_sizes,computation_times=computation_times, time_stamps=time_stamps,joint_traj=joint_traj,joint_vel_traj=joint_vel_traj,eef_traj=eef_traj, x_traj=x_traj,dx_traj=dx_traj,ddx_traj=ddx_traj,ins=ins,pos_init=initial_position,pos_goal=goal_position,angle_swap=angle_swap,sol_id=sol_id,sol_id_req=req.sol_id,traj_angle=traj_angle,gripper_length=self.gripper_length,gripper_width=self.gripper_width,gripper_depth=self.gripper_depth,ins_offsets=self.ins_offsets,drop_height_offset=self.get_parameter('drop_height_offset').value)
        return res

    def initialize_dmp(self, trajectory_file, n_dmps=3, n_bfs=10, K=25, rescale='rotodilatation', basis='rbf', alpha_s=4.0, tol=None):
        if tol is None:
            tol = self.tol
        self.n_bfs_p = n_bfs + 1 #dmp library adds one to the specified number of basis functions
        trajectory = np.loadtxt(trajectory_file, delimiter=',')  # Adjust the delimiter if necessary
        T = trajectory[-1, 0] - trajectory[0, 0]
        n_traj_steps, _ = trajectory.shape
        dt = T/n_traj_steps
        t_traj = trajectory[:, 0]
        xs_traj = trajectory[:, 1:1 + n_dmps]
        xds_traj = trajectory[:, 1 + n_dmps:1 + 2 * n_dmps]
        xdds_traj = trajectory[:, 1 + 2 * n_dmps:1 + 3 * n_dmps]
        x_0 = xs_traj[0, :] # + self.init_pertub
        x_goal = xs_traj[-1, :] # + self.goal_pertub
        x_0 = np.zeros(n_dmps) # + self.init_pertub
        x_goal = np.zeros(n_dmps) # + self.goal_pertub
        #self.learned_position = x_goal - x_0
        #self.learned_L = np.linalg.norm(self.learned_position)
        #self.stack_limits_z = [self.learned_L*self.stack_z_ratio,self.learned_L-self.learned_L*self.stack_z_ratio]
        #self.n_time_steps, _ = trajectory.shape
        dmp_out = dmp.DMPs_cartesian(
            n_dmps = n_dmps,
            n_bfs = n_bfs,
            #dt = dt,
            x_0 = x_0,
            x_goal = x_goal,
            #T = T,
            K = K,
            rescale = rescale,
            alpha_s = alpha_s,
            tol = tol,
            basis=basis)

        dmp_out.imitate_path(x_des=xs_traj,dx_des=xds_traj,ddx_des=xdds_traj, t_des =t_traj)

        return dmp_out

    def get_new_weights(self,dmp,model,ins,avoidance_angle):
        ins_torch = torch.tensor(ins, dtype=torch.float32)
        with torch.no_grad():
            new_means = model(ins_torch)

        new_means = np.reshape(new_means, (self.dmp.n_bfs+1, 2))
        #f = np.linalg.norm(dist)/self.dmp_L_demo #replace 0.15 with the distance between x_init and x_end

        #modify the means depending on the avoidance angle:
        self.dmp.w[1, :] = new_means[:, 0]
        self.dmp.w[2, :] = np.cos(avoidance_angle) * new_means[:, 1]
        self.dmp.w[0, :] = np.sin(avoidance_angle) * new_means[:, 1]

        return dmp,new_means

    def generate_three_segment_trajectory(self,i, g, in1, in2, in3, perp_direction, num_points=100):
        i = np.array(i, dtype=np.float64)
        g = np.array(g, dtype=np.float64)

        direction = g - i
        dist = np.linalg.norm(direction)
        if dist == 0:
            raise ValueError("Initial and goal points are the same.")

        direction_unit = direction / dist

        # Base perpendicular direction from user input
        if perp_direction == 0:
            base_perp = np.array([0, 1, 0])  # +Y
        elif perp_direction == 1:
            base_perp = np.array([0, -1, 0])  # -Y
        elif perp_direction == 2:
            base_perp = np.array([0, 0, 1])  # +Z
        else:
            raise ValueError("perp_direction must be 0 (neg_y), 1 (pos_y), or 2 (pos_z)")

        # If base_perp is parallel to direction, switch base vector
        if np.allclose(np.cross(direction_unit, base_perp), 0):
            base_perp = np.array([1, 0, 0])  # fallback to +X

        # Compute true perpendicular vector in the plane orthogonal to direction
        perp_unit = np.cross(direction_unit, np.cross(base_perp, direction_unit))
        perp_unit /= np.linalg.norm(perp_unit)

        # Intermediate points
        p1 = i + in2 * direction + in1 * dist * perp_unit
        p2 = i + in3 * direction + in1 * dist * perp_unit

        # Segment lengths
        l1 = np.linalg.norm(p1 - i)
        l2 = np.linalg.norm(p2 - p1)
        l3 = np.linalg.norm(g - p2)
        total_length = l1 + l2 + l3

        # Number of points per segment
        n1 = max(2, int(num_points * (l1 / total_length)))
        n2 = max(2, int(num_points * (l2 / total_length)))
        n3 = num_points - (n1 + n2) + 1  # Ensure total points == num_points

        def interpolate(a, b, n):
            return [a + t * (b - a) for t in np.linspace(0, 1, n, endpoint=False)]

        segment1 = interpolate(i, p1, n1)
        segment2 = interpolate(p1, p2, n2)
        segment3 = interpolate(p2, g, n3)

        trajectory = segment1 + segment2 + segment3 + [g]
        return np.array(trajectory)

    def create_pose_trajectory(self,position_trajectory, constant_quaternion):
        """
        Create a Pose[] trajectory from a position trajectory and a constant quaternion.

        Args:
            position_trajectory (list of tuple): List of (x, y, z) positions.
            constant_quaternion (Quaternion): Constant quaternion (x, y, z, w).

        Returns:
            list of geometry_msgs.msg.Pose: Pose[] trajectory.
        """
        pose_trajectory = []

        for position in position_trajectory:
            pose = Pose()
            # Set position
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
            # Set orientation (constant quaternion)
            pose.orientation = constant_quaternion

            pose_trajectory.append(pose)

        return pose_trajectory

    def create_pose_trajectory_moveit(self,position_trajectory, constant_quaternion):
        """
        Create a Pose[] trajectory from a position trajectory and a constant quaternion.

        Args:
            position_trajectory (list of tuple): List of (x, y, z) positions.
            constant_quaternion (Quaternion): Constant quaternion (x, y, z, w).

        Returns:
            list of geometry_msgs.msg.Pose: Pose[] trajectory.
        """
        pose_trajectory = []
        #constant_quaternion.x = -constant_quaternion.x
        #constant_quaternion.y = -constant_quaternion.y
        for position in position_trajectory:
            pose = Pose()
            # Set position
            pose.position.x = -position[0]
            pose.position.y = -position[1]
            pose.position.z = position[2]
            # Set orientation (constant quaternion)
            pose.orientation.x = -constant_quaternion.x
            pose.orientation.y = constant_quaternion.y
            pose.orientation.z = constant_quaternion.z
            pose.orientation.w = constant_quaternion.w
            #self.get_logger().info(f'constant quaternion: {pose.orientation}')
            pose_trajectory.append(pose)

        return pose_trajectory

    def trajectory_complete_callback(self, msg: Bool):
        if msg.data:
            self.trajectory_completed = True
            self.get_logger().info('Trajectory finished!')
        else:
            self.trajectory_completed = False
            self.get_logger().info('Trajectory planned, executing...')

    def wait_for_trajectory_completion(self):
        eef_traj = []
        time_stamps = []
        joint_traj = []
        joint_vel_traj = []
        self.get_logger().info("Waiting for trajectory completion...")
        start_time = time.time()
        time_counter = 0
        # Wait for the flag to be set by the callback
        while rclpy.ok() and not self.trajectory_completed:
            eef_traj.append([self.eef_pose.position.x,self.eef_pose.position.y,self.eef_pose.position.z,self.eef_pose.orientation.x,self.eef_pose.orientation.y,self.eef_pose.orientation.z,self.eef_pose.orientation.w])
            joint_traj.append(self.joint_states)
            joint_vel_traj.append(self.joint_vels)
            time_stamps.append(time.time())
            #rclpy.spin_once(self, timeout_sec=0.1)  # Process incoming messages with a timeout
            if (time.time()-start_time)>1.0:
                start_time = time.time()
                time_counter+=1
                self.get_logger().info(f"#{time_counter} still waiting...")
                if time_counter > 20:
                    self.get_logger().info(f"Timeout reached, abort!")
                    eef_traj = [-1]
                    time_stamps = [-1]
                    joint_traj = [-1]
                    joint_vel_traj = [-1]
                    break
            time.sleep(0.01)  # Simple sleep to control loop frequency, otherwise run as fast as possible

        if self.trajectory_completed:
            self.get_logger().info("Trajectory completed successfully.")
            self.trajectory_completed = None  # Reset the flag for subsequent waits
        else:
            self.get_logger().warn(f"Stopped waiting due to node shutdown or failure: {self.trajectory_completed}")
        return np.array(eef_traj), np.array(time_stamps), np.array(joint_traj), np.array(joint_vel_traj)

    def attach_cube(self,name='cube'):
        #get cube pose
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for attach service...')

        request = Attach.Request()
        request.model_name_1 = name
        request.link_name_1 = "link"
        request.model_name_2 = "ur"
        request.link_name_2 = "wrist_3_link"
        result = self.attach_client.call(request)
        #self.get_logger().info(f'Cube {cube_id} attched!')

    def detach_cube(self,name='cube'):
        #get cube pose
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for detach service...')

        request = Attach.Request()
        request.model_name_1 = name
        request.link_name_1 = "link"
        request.model_name_2 = "ur"
        request.link_name_2 = "wrist_3_link"
        result = self.detach_client.call(request)



def main(args=None):
    rclpy.init(args=args)
    node = HighLevelControlNode()
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
