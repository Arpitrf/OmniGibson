import os
import yaml
import  pdb
import pickle

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory

temp_prior = th.tensor([
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.,   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.2,   -0.,    -0.,    -0.,     -1.,   ],
    [ 0.   ,  0.,     0.,     0.007, -0.001,  0.207, -0.028, -0.105, -0.015,  -1.,   ],
    # [ 0.   ,  0.,    -1.522,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
    [ 0.   ,  0.,    -1.222,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
    [ 0.527,  0.026,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
    [ 0.   ,  0.,    -0.089,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
    # [ 0.   ,  0.,     0.,     0.191, -0.052,  0.028,  0.029, -0.384, -0.44,   -1.,   ],
    [ 0.   ,  0.,     0.,     0.131, -0.052,  0.068,  0.029, -0.384, -0.44,   -1.,   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.,   ]
])

def add_noise(temp_prior):
    traj_length = len(temp_prior)
    for i in range(1, traj_length-1):
        if i in [1, 2, 6]:
            # add noise to arm
            noise = th.from_numpy(np.random.uniform(-0.05, 0.05, 3))
            temp_prior[i, 3:6] += noise
        elif i in [3]:
            # add noise to base
            # noise_base_xy = np.random.uniform(-0.1, 0.1, 2)
            noise_base_xy = [0.0, 0.0]
            noise_base_yaw = np.random.uniform(-0.2, 0.2)
            # pdb.set_trace()
            temp_prior[i, 0:3] += th.tensor([noise_base_xy[0], noise_base_xy[1], noise_base_yaw])
    return temp_prior

def move_primitive(action):
    current_pose = robot.get_relative_eef_pose(arm='right')
    current_pos = current_pose[0]
    current_orn = current_pose[1]
    
    print("action: ", action)
    delta_pos = action[3:6]
    print("delta_pos: ", delta_pos)
    delta_orn = action[6:9]
    grasp_action = action[9]
    
    target_pos = current_pos + delta_pos
    print("current_pos, target_pos: ", current_pos, target_pos)
    # print("type(target_pos): ", type(target_pos))
    target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
    # print("target_orn: ", target_orn, target_orn.as_quat())
    target_orn = th.tensor(target_orn.as_quat())

    target_pose = (target_pos, target_orn)
    # print("current_pose: ", current_pose)
    # print("target_pose: ", target_pose)

    obs, info = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                            stop_on_contact=False,
                                                                            ignore_failure=True,
                                                                            stop_if_stuck=False), grasp_action)
    

    # obtain target pose2d
    current_base_pos, current_base_orn_quat = robot.get_position_orientation()
    current_base_yaw = R.from_quat(current_base_orn_quat).as_euler('XYZ')[2]

    # print("action: ", action)
    delta_base_pos = action[0:2] # this is in the robot frame
    # conver delta pos from robot frame to world frame
    robot_to_world = np.eye(4)
    robot_to_world[:3, :3] = R.from_quat(current_base_orn_quat).as_matrix()
    robot_to_world[:3, 3] = np.transpose(np.array([0.0, 0.0, 0.0]))
    delta_base_pos_homo = np.array([delta_base_pos[0], delta_base_pos[1], 0.0, 1.0])
    delta_base_pos_world = np.dot(robot_to_world, delta_base_pos_homo)
    delta_base_pos_world = th.from_numpy(delta_base_pos_world)
    delta_base_yaw = action[2]
    # # remove later
    # delta_base_yaw = 0.78

    # target_base_pos = current_base_pos + delta_base_pos_world[:3]
    target_base_pos = current_base_pos + th.tensor([delta_base_pos[0], delta_base_pos[1], 0.0])
    target_base_yaw = current_base_yaw + delta_base_yaw
    target_pose2d = th.tensor([target_base_pos[0], target_base_pos[1], target_base_yaw])
    print("current_base_pos, delta_base_pos: ", current_base_pos[:2], delta_base_pos_world[:2])
    print("current_base_yaw, delta_base_yaw: ", current_base_yaw, delta_base_yaw)
    obs, info = execute_controller(action_primitives._navigate_to_pose_direct(target_pose2d), grasp_action)


    # Hack to ensure that even if primitive does not return any action (if delta pose is 0), grasp action is performed
    action = action_primitives._empty_action()
    obs, info = execute_controller([action], grasp_action)

    return obs, info

def execute_controller(ctrl_gen, grasp_action):
    obs, info = env.get_obs()
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[20] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
    return obs, info

def perform_grasp():
    # ======================= Move hand to grasp pose ================================    
    grasp_action = 1.0
    # w.r.t world
    # target_pose = (th.tensor([0.1829, 0.4876, 0.4051]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    # w.r.t robot
    target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), th.tensor([-0.0256,  0.0228,  0.6444,  0.7640]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                       grasp_action) 
    for _ in range(40):
        og.sim.step()
    current_pose_world = robot.eef_links["right"].get_position_orientation()
    print("move hand down completed. Desired and Reached right eef pose reached: ", target_pose[0], current_pose_world[0])
    # =================================================================================

def correct_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=100.0,
        dynamic_friction=100.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset(head_joints_pos=head_joints)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=1)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, 0.0]
box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
config["objects"] = [
    {
        "type": "DatasetObject",
        "name": "shelf",
        "category": "shelf",
        "model": "eniafz",
        "position": [1.5, 0, 1.0],
        "scale": [2.0, 2.0, 1.0],
        "orientation": rot_quat,
    },   
    {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "fqluyq",
        "position": [0, 0.6, 0.3],
        "orientation": [0, 0, 0, 1]
    },
    {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
correct_gripper_friction()
obj = env.scene.object_registry("name", "box")
obj.root_link.mass = 1e-2
traj_length = len(temp_prior)

# add noise to the original trajectory
# temp_prior = add_noise(temp_prior)

for _ in range(10):
    custom_reset(env, robot)
    # grasp
    perform_grasp()
    temp_prior = add_noise(temp_prior)
    for t in range(traj_length):
        # replay the traj
        move_primitive(temp_prior[t])
        for _ in range(50):
            og.sim.step()

# # Initializations
# obj_fn = ClassMismatchError(key="gripper_obj_seg", weight=1)
# CEM_actions, CEM_noises, CEM_rewards = [], [], []
# num_samples = 200
# n_ctxt = 1
# K = 12
# mu_x = np.zeros(TRAJ_LENGTH)  # Example: 2-dimensional problem
# sigma_x = np.eye(TRAJ_LENGTH) * 0.005  # Initial covariance (identity matrix)
# mu_y = np.zeros(TRAJ_LENGTH)  # Example: 2-dimensional problem
# sigma_y = np.eye(TRAJ_LENGTH) * 0.005  # Initial covariance (identity matrix)
# mu_z = np.zeros(TRAJ_LENGTH)  # Example: 2-dimensional problem
# sigma_z = np.eye(TRAJ_LENGTH) * 0.005  # Initial covariance (identity matrix)