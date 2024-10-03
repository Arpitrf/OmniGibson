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

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory

# def get_pose_wrt_robot():
#      # obtain target pose w.r.t robot
#     target_pose = np.eye(4)
#     target_pose[:3, :3] = R.from_quat(place_pose[1]).as_matrix()
#     target_pose[:3, 3] = np.transpose(place_pose[0])
#     robot_pose = robot.get_position_orientation()
#     robot_to_world = np.eye(4)
#     robot_to_world[:3, :3] = R.from_quat(robot_pose[1].numpy()).as_matrix()
#     robot_to_world[:3, 3] = np.transpose(robot_pose[0].numpy())

#     target_pose_wrt_robot = np.dot(np.linalg.inv(robot_to_world), target_pose) 
    
#     target_pos = target_pose_wrt_robot[:3, 3]
#     target_orn = np.array(R.from_matrix(target_pose_wrt_robot[:3, :3]).as_quat())
#     target_pose = (th.from_numpy(target_pos), th.from_numpy(target_orn))
#     print("target_pos: ", target_pos, target_orn)

def dump_to_memory(env, robot, episode_memory, number_of_collisions=0):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    for k in obs['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs['robot0']['robot0:eyes:Camera:0'][k].cpu().numpy())
    # # add gripper+object seg
    # gripper_obj_seg = obtain_gripper_obj_seg(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'], obs_info['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
    # episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info['robot0']['robot0:eyes:Camera:0'][k])


    is_grasping = robot.custom_is_grasping()
    box = env.scene.object_registry("name", "box")
    # is_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
    is_in_collision = False
    if number_of_collisions > 5:
        is_in_collision = True
    print("is_in_collision: ", number_of_collisions, is_in_collision)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_in_collision)

# def custom_reset(env, robot, episode_memory): 
#     scene_initial_state = env.scene._initial_state
    
#     base_yaw = 90
#     r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
#     r_quat = R.as_quat(r_euler)
#     scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    
#     head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

#     # Reset environment and robot
#     env.reset()
#     robot.reset(head_joints_pos=head_joints)

#     # Step simulator a few times so that the effects of "reset" take place
#     for _ in range(10):
#         og.sim.step()

#     # add to memory
#     dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, gripper_closed, episode_memory=None):
    number_of_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory, number_of_collisions) 
            number_of_collisions = 0
            continue
        if gripper_closed:
            # if left hand is ik
            # action[18] = -1
            # if left hand is joint controller
            action[20] = -1
        else: 
            # action[18] = 1
            action[20] = 1
        # print("action: ", action[:3], action[14:17])
        env.step(action)

        # debugging:
        box = env.scene.object_registry("name", "box")
        is_contact = detect_robot_collision_in_sim(robot, filter_objs=[box])
        if is_contact:
            number_of_collisions += 1
            # print("Collided! number_of_collisions: ", number_of_collisions)
        # print("is_arm_in_collision: ", is_contact)
        # current_pos_world = robot.eef_links["right"].get_position_orientation()
        # print("current_pose_world: ", current_pos_world[0])


def primitive(episode_memory):
    # ======================= Move hand to place pose ================================
    gripper_closed = True
    # w.r.t world
    # place_pose =  (np.array([ 1.1888, -0.1884,  0.8387]), np.array([-0.0489, -0.0063,  0.5555,  0.8301]))
    # w.r.t robot
    place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))
    # # add noise to place pos
    place_pos = place_pose[0]
    place_orn = place_pose[1]
    place_noise_x, place_noise_y, place_noise_z = np.random.uniform(-0.1, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.3, 0.3)
    place_noise = th.tensor([place_noise_x, place_noise_y, place_noise_z])
    place_pos += place_noise
    place_pose = (place_pos, place_orn)
    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=False, episode_memory=episode_memory, gripper_closed=gripper_closed), 
                       env, 
                       robot, 
                       gripper_closed,
                       episode_memory)
    current_pose_world = robot.eef_links["right"].get_position_orientation()
    current_pose_robot = robot.get_relative_eef_pose(arm='right')
    print("move hand to place location completed. Desired and Reached right eef pose reached: ", place_pose[0], current_pose_robot[0])
    # ====================================================================================

    #TODO: Add a 0 action here

    # # ============= Open grasp =================
    # gripper_closed = False
    # action = action_primitives._empty_action()
    # # if left hand is IK
    # # action[18] = -1
    # # if left has is joint controller
    # action[20] = 1
    # execute_controller([action], env, robot, gripper_closed, episode_memory)
    # # step the simulator a few steps to let the gripper close completely
    # for _ in range(40):
    #     og.sim.step()
    # # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # episode_memory.add_action('actions', action_to_add)
    # # ==========================================

def randomize_robot():
    # for manipulation
    # base_pose = robot.get_position_orientation()
    # base_pos = base_pose[0]
    # base_x_noise = np.random.uniform(-0.1, 0.05)
    # base_y_noise = np.random.uniform(-0.1, 0.05)
    # base_noise = th.tensor([base_x_noise, base_y_noise, 0.0])
    # base_noise = th.tensor([-0.1, 0.0, 0.0])
    # base_pos += base_noise 
    # scene_initial_state['object_registry']['robot0']['root_link']['pos'] = base_pos
    
    # base_yaw = R.from_quat(base_pose[1]).as_euler('XYZ', degrees=True)[2]
    # print("base_yaw: ", base_yaw)
    # base_yaw_noise = np.random.uniform(-15, 15)
    # # remove later
    # base_yaw_noise = 45
    # base_yaw += base_yaw_noise
    # r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    # r_quat = R.as_quat(r_euler)
    # scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    # print("r_quat: ", r_quat)

    # robot.set_position_orientation(base_pos, r_quat)

    action = th.zeros(robot.action_dim)
    action[20] = -1
    print("action: ", action)
    base_x_vel = np.random.uniform(-0.1, 0.05)
    base_y_vel = np.random.uniform(-0.1, 0.1)
    base_yaw_vel = np.random.uniform(-0.2, 0.2)
    # action[:3] = th.tensor([0.0, 0.0, 0.2])
    action[:3] = th.tensor([base_x_vel, base_y_vel, base_yaw_vel])
    env.step(action)
    timesteps = np.random.randint(15, 30)
    for _ in range(timesteps): # was 30 before
        og.sim.step()
    
    action = th.zeros(robot.action_dim)
    action[20] = -1
    env.step(action)

    # Randomizing head pose
    # default_head_joints = np.array([-0.20317451, -0.7972661])
    default_head_joints = np.array([-0.5031718015670776, -0.9972541332244873])
    noise_1 = np.random.uniform(-0.1, 0.1, 1)
    noise_2 = np.random.uniform(-0.1, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints_pos = default_head_joints + noise
    # head_joints_pos = np.array([0.0, -0.5])
    head_joints_pos = th.from_numpy(head_joints_pos)
    head_joints_pos = th.tensor(head_joints_pos, dtype=th.float32)
    robot.set_joint_positions(head_joints_pos, indices=robot.camera_control_idx)

    # add to memory
    dump_to_memory(env, robot, episode_memory)

    # for _ in range(50):
    #     og.sim.step()
    # # robot.reset(head_joints_pos=head_joints)

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=5)
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
    }
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
print(robot.name)
# pdb.set_trace()

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

# og.clear()
og.sim.restore(["episode_00000_before_place.json"])

# getting all objects in scene
# env.scene.objects

box = env.scene.object_registry("name", "box")
box.root_link.mass = 1e-2
print("box.mass: ", box.mass)

# # debugging
# print("robot left gripper left finger: ", robot.finger_links['left'][0].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())
# print("robot left gripper right finger: ", robot.finger_links['left'][1].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

save_folder = 'place_in_shelf_data_test2'
os.makedirs(save_folder, exist_ok=True)

# Obtain the number of episodes
episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)

# # save the start simulator state
# og.sim.save(f'{save_folder}/episode_{episode_number:05d}_start.json')
# arr = scene.dump_state(serialized=True)
# with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
#     pickle.dump(arr, f)
            
for _ in range(100):
    og.sim.step()

state = og.sim.dump_state(serialized=False)
for i in range(50):
    print(f"---------------- Episode {i} ------------------")
    episode_memory = Memory()
    
    # randomize base pose and head pose a bit
    randomize_robot()
    
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    primitive(episode_memory)
    episode_memory.dump(f'{save_folder}/dataset.hdf5')
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    
    og.sim.load_state(state, serialized=False)
    
    # remove later
    for _ in range(30):
        og.sim.step()

    del episode_memory
    episode_number += 1

# # save the end simulator state
# og.sim.save(f'{save_folder}/episode_{episode_number:05d}_end.json')
# arr = scene.dump_state(serialized=True)
# with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
#     pickle.dump(arr, f)


# Always shut down the environment cleanly at the end
# og.clear()
og.shutdown()



