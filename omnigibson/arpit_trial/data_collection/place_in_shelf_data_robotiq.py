import os
import yaml
import  pdb
import pickle
import h5py

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.arpit_trial.data_collection.TARGET_EEF_POSES as TEP

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.objects import PrimitiveObject
from omnigibson.arpit_trial.memory import Memory
from omnigibson.utils.python_utils import nums2array
from omnigibson import object_states

def set_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=10.0,
        dynamic_friction=10.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def initialize_robot():
    # robot.set_joint_positions(positions=th.tensor([0.2]), indices=robot.gripper_control_idx["right"])
    # for _ in range(100):
    #     og.sim.step()
    # Ensure hand is at the held pose
    grasp_action = 1.0
    current_eef_pose = robot.get_relative_eef_pose(arm='right')
    target_pose = (th.tensor(TEP.HELD_POS["robot"], dtype=th.float32), current_eef_pose[1])
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action=grasp_action)

def set_extrinsic_matrix(robot, camera_link="xtion_link"):
    # Alternative approach using robot's built-in methods
    camera_link = robot.links["xtion_optical_frame"]
    base_link = robot.links["base_footprint"]

    # Get world poses
    camera_pos, camera_quat = camera_link.get_position_orientation()
    base_pos, base_quat = base_link.get_position_orientation()

    camera_mat = T.pose2mat((camera_pos, camera_quat))
    base_mat = T.pose2mat((base_pos, base_quat))
    camera_to_base = th.linalg.inv(base_mat) @ camera_mat

    robot._extrinsic_matrix = camera_to_base


def dump_to_memory(env, robot, episode_memory, number_of_collisions=0, reached_singularity=False):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    proprio['extrinsic_matrix'] = robot._extrinsic_matrix
    # breakpoint()
    # convert tuple to tensor
    xtion_rgb_optical_frame_pose =  robot.links["xtion_rgb_optical_frame"].get_position_orientation()
    xtion_depth_optical_frame_pose =  robot.links["xtion_depth_optical_frame"].get_position_orientation()
    proprio['xtion_rgb_optical_frame'] = th.cat((xtion_rgb_optical_frame_pose[0], xtion_rgb_optical_frame_pose[1]))
    proprio['xtion_depth_optical_frame'] = th.cat((xtion_depth_optical_frame_pose[0], xtion_depth_optical_frame_pose[1]))
   
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    robot_name = env.robots[0].name
    for k in obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k].cpu().numpy())
    # # add gripper+object seg
    # gripper_obj_seg = obtain_gripper_obj_seg(obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'], obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'])
    # episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k])


    is_grasping = robot.custom_is_grasping()
    is_in_collision = False
    if number_of_collisions > 5:
        is_in_collision = True
    print("is_in_collision: ", number_of_collisions, is_in_collision)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_in_collision)
    episode_memory.add_extra('singularities', reached_singularity)


def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    number_of_collisions = 0
    singularities = []
    reached_singularity = False
    for action in ctrl_gen:
        if action == 'Done':
            print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(action_primitives.move_hand_direct_ik_orn_error))
            
            if episode_memory is not None:      
                dump_to_memory(env, robot, episode_memory, number_of_collisions, reached_singularity=reached_singularity) 
            number_of_collisions = 0
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action[:3], action[14:17])
        env.step(action)

        # # ============================================= Check for collisions =============================================
        # # Check if robot right gripper fingers are in collision
        # box = env.scene.object_registry("name", "box")
        # gripper_fingers_is_contact = detect_robot_collision_in_sim(robot, filter_objs=[box])

        # # Check if robot right gripper is in collision
        # gripper_is_contact = False
        # # TODO: Remove hardcoding from link names
        # gripper_links = ["gripper_right_link"]
        # for gripper_link in gripper_links:
        #     lis = robot.links[gripper_link].contact_list()
        #     if len(lis) > 0:
        #         # print(f"arm_right_{j}_link in contact at step {i}: ", lis)
        #         gripper_is_contact = True

        # # Check if robot right arm is in collision
        # robot_is_contact = False
        # # TODO: Remove hardcoding from indices
        # for j in range(1,8):
        #     lis = robot.links[f"arm_right_{j}_link"].contact_list()
        #     if len(lis) > 0:
        #         # print(f"arm_right_{j}_link in contact at step {i}: ", lis)
        #         robot_is_contact = True

        # # Check if box is in collision
        # box_is_contact = False
        # box_contact_bodies = list(box.states[ContactBodies].get_value())
        # # two fingers are already in contact with the box 
        # if len(box_contact_bodies) > 2:
        #     box_is_contact = True
        #     # print("box_contact_bodies: ", box_contact_bodies)

        # is_contact = robot_is_contact or box_is_contact or gripper_is_contact or gripper_fingers_is_contact
        # if is_contact:
        #     number_of_collisions += 1
        #     # print("Collided! number_of_collisions: ", number_of_collisions)
        # # ====================================================================================

        # # if singularity is reached in this episode, do not add to memory
        # singularity = robot._controllers["arm_right"].singularity
        # singularities.append(singularity)

        # if sum(singularities) > 3:
        #     reached_singularity = True
        #     # remove the last action from memory
        #     # episode_memory.data['actions']['actions'].pop()
        #     # return


def primitive(episode_memory):
    # ======================= Move hand to place pose ================================
    curr_eef_pose = robot.get_relative_eef_pose(arm='right')
    curr_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    grasp_action = 1.0
    place_pose = (PLACE_POS, curr_eef_pose[1])
    # w.r.t world
    # place_pose =  (th.tensor([ 1.1688, -0.1884,  0.8387]), curr_eef_pose_world[1])
    # w.r.t robot
    # place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), curr_eef_pose[1])

    # # add noise to place pos
    # place_pos = place_pose[0]
    # place_orn = place_pose[1]
    # # Noise range 1 (0-300)
    # place_noise_x, place_noise_y, place_noise_z = np.random.uniform(0.0, 0.15), np.random.uniform(-0.17, 0.17), np.random.uniform(-0.1, 0.02) # np.random.uniform(-0.1, 0.05)
    # # Noise range 2 (300-600)
    # # place_noise_x, place_noise_y, place_noise_z = np.random.uniform(-0.15, 0.15), np.random.uniform(-0.17, 0.17), np.random.uniform(-0.2, 0.15) # np.random.uniform(-0.1, 0.05)
    # # Noise range 3 (600-900)
    # # place_noise_x, place_noise_y, place_noise_z = np.random.uniform(-0.25, 0.15), np.random.uniform(-0.25, 0.25), np.random.uniform(-0.3, 0.2) # np.random.uniform(-0.1, 0.05)
    # place_noise = th.tensor([place_noise_x, place_noise_y, place_noise_z])
    # place_pos += place_noise
    # place_pose = (place_pos, place_orn)    

    # # trying out the sampling pose function provided by OG
    # place_pose = action_primitives._sample_pose_with_object_and_predicate(object_states.Inside, held_obj=box, target_obj=shelf)
    # curr_robot_pos_world, curr_robot_orn_world = robot.get_position_orientation()
    # curr_robot_pose_world = th.eye(4)
    # curr_robot_pose_world[:3, :3] = th.tensor(R.from_quat(curr_robot_orn_world).as_matrix())
    # curr_robot_pose_world[:3, 3] = curr_robot_pos_world
    # marker_pos = curr_robot_pose_world @ th.tensor([*place_pose[0], 1])
    # eef_marker.set_position_orientation(position=marker_pos[:3])

    eef_marker.set_position_orientation(position=place_pose[0])

    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory, grasp_action=grasp_action), 
                       env, 
                       robot, 
                       grasp_action,
                       episode_memory)

    # Debugging
    post_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose_world[0] - place_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose_world[1], place_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # breakpoint()
    # ====================================================================================


    # ============= Open grasp =================
    robot.set_joint_positions(positions=th.tensor([0.2]), indices=robot.gripper_control_idx["right"])
    for _ in range(100):
        og.sim.step()

    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # execute_controller([action], env, robot, grasp_action)
    # # step the simulator a few steps to let the gripper close completely
    # for _ in range(100):
    #     og.sim.step()
    # # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # episode_memory.add_action('actions', action_to_add)
    # ==========================================

def randomize_robot():
    
    # Randomize initial hand pose
    grasp_action=1.0
    current_eef_pose = robot.get_relative_eef_pose(arm='right') 
    # Noise range 1 (0-300)
    noise_x, noise_y, noise_z = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.05)
    # Noise range 2 (300-600)
    # noise_x, noise_y, noise_z = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)
    # Noise range 3 (600-900)
    # noise_x, noise_y, noise_z = np.random.uniform(-0.1, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.2, 0.2)
    noise = th.tensor([noise_x, noise_y, noise_z])
    # breakpoint()
    target_pose = (th.cat((current_eef_pose[0][:2] + noise[:2], PLACE_POS[2:] + noise[2:])), current_eef_pose[1])
    # Ensure the target pose is within the robot's reachable workspace
    reachable_workspace = robot.arm_reachable_workspace["right"]
    target_pose = (th.clamp(target_pose[0], min=reachable_workspace["min"], max=reachable_workspace["max"]), target_pose[1])
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action=grasp_action)
    for _ in range(20):
        og.sim.step()
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    
    # Randomize initial base pose
    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = 1
    # base_x_vel = np.random.uniform(-0.01, 0.05)
    base_x_vel = np.random.uniform(0.08, 0.13)
    base_y_vel = np.random.uniform(-0.1, 0.1)
    base_yaw_vel = np.random.uniform(-0.01, 0.01) # 0.2 originally
    # action[:3] = th.tensor([0.0, 0.0, 0.2])
    action[:3] = th.tensor([base_x_vel, base_y_vel, base_yaw_vel])
    env.step(action)
    timesteps = np.random.randint(15, 40)
    for _ in range(timesteps):
        og.sim.step()
    
    # Randomize head pose
    default_head_joints = np.array([-0.503, -0.797])
    noise_1 = np.random.uniform(-0.05, 0.05, 1)
    noise_2 = np.random.uniform(-0.05, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints_pos = default_head_joints + noise
    head_joints_pos = th.from_numpy(head_joints_pos)
    head_joints_pos = th.tensor(head_joints_pos, dtype=th.float32)
    robot.set_joint_positions(head_joints_pos, indices=robot.camera_control_idx)

    # Loop a few physics steps to let the robot settle
    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = 1
    env.step(action)
    for _ in range(50):
        og.sim.step()

    # Set the extrinsic matrix
    set_extrinsic_matrix(robot)

    # # add to memory
    # dump_to_memory(env, robot, episode_memory)

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
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    }
]

env = og.Environment(configs=config)
save_folder = 'place_in_shelf_data_final'
os.makedirs(save_folder, exist_ok=True)

og.sim.restore(["place_in_shelf_final/episode_00000_place_start_osc.json"])

scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

set_gripper_friction()

# getting all objects in scene
# env.scene.objects
shelf = env.scene.object_registry("name", "shelf")
shelf.root_link.mass = 1e3
box = env.scene.object_registry("name", "box")
box.root_link.mass = 1e-1
# print("box.mass: ", box.mass)
eef_marker = env.scene.object_registry("name", "marker")

# set control specific parameters
robot.controllers["arm_right"].kp[:3] = nums2array(nums=150, dim=3, dtype=th.float32)
robot.controllers["arm_right"].kp[-3:] = nums2array(nums=150, dim=3, dtype=th.float32)
robot.controllers["gripper_right"]._motor_type = "velocity"
# robot.controllers["gripper_right"].kp = 1000
robot.controllers["gripper_right"]._use_impedances = True
# robot.controllers["gripper_right"].kp = 0.0
# robot.controllers["gripper_right"].kd = 0.0

# # debugging
# print("robot left gripper left finger: ", robot.finger_links['left'][0].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())
# print("robot left gripper right finger: ", robot.finger_links['left'][1].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())

# Obtain the number of episodes
episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)

for _ in range(10):
    og.sim.step()

# initialize robot
PLACE_POS = th.tensor(TEP.PLACE_POS_SHELF["world"], dtype=th.float32)
# breakpoint()
initialize_robot()


state = og.sim.dump_state(serialized=False)
for i in range(600):
    print(f"---------------- Episode {episode_number} ------------------")
    episode_memory = Memory()
    
    # Randomize base pose and head pose a bit
    randomize_robot()
    breakpoint()
    
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    primitive(episode_memory)
    # episode_memory.dump(f'{save_folder}/dataset.hdf5')
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    
    for _ in range(10):
        og.sim.step()
    og.sim.load_state(state, serialized=False)
    for _ in range(10):
        og.sim.step()

    del episode_memory
    episode_number += 1

# Always shut down the environment cleanly at the end
# og.clear()
og.shutdown()


