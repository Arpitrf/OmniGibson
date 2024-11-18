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
import omnigibson.utils.transform_utils as T
from memory import Memory

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
    box = env.scene.object_registry("name", "box")
    # is_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
    is_in_collision = False
    if number_of_collisions > 5:
        is_in_collision = True
    print("is_in_collision: ", number_of_collisions, is_in_collision)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_in_collision)
    episode_memory.add_extra('singularities', reached_singularity)
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

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    number_of_collisions = 0
    singularities = []
    reached_singularity = False
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory, number_of_collisions, reached_singularity=reached_singularity) 
            number_of_collisions = 0
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action[:3], action[14:17])
        env.step(action)

        # debugging:
        box = env.scene.object_registry("name", "box")
        is_contact = detect_robot_collision_in_sim(robot, filter_objs=[box])
        if is_contact:
            number_of_collisions += 1
            # print("Collided! number_of_collisions: ", number_of_collisions)

        # if singularity is reached in this episode, do not add to memory
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)

        if sum(singularities) > 3:
            reached_singularity = True
            # remove the last action from memory
            # episode_memory.data['actions']['actions'].pop()
            # return


def primitive(episode_memory):
    # ======================= Move hand to place pose ================================
    grasp_action = -1.0
    # w.r.t world
    # place_pose =  (np.array([ 1.1888, -0.1884,  0.8387]), np.array([-0.0489, -0.0063,  0.5555,  0.8301]))
    # w.r.t robot
    place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))

    # # move the right eef 10 cm forward
    # current_eef_pose = robot.get_relative_eef_pose(arm='right')
    # # current_eef_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
    # print("current_eef_pose: ", current_eef_pose)
    # place_pose = (current_eef_pose[0] + th.tensor([0.1, 0.0, 0.0]), current_eef_pose[1])

    # add noise to place pos
    place_pos = place_pose[0]
    place_orn = place_pose[1]
    place_noise_x, place_noise_y, place_noise_z = np.random.uniform(0.0, 0.15), np.random.uniform(-0.17, 0.17), np.random.uniform(-0.1, 0.02) # np.random.uniform(-0.1, 0.05)
    place_noise = th.tensor([place_noise_x, place_noise_y, place_noise_z])
    place_pos += place_noise
    place_pose = (place_pos, place_orn)    
    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=False, episode_memory=episode_memory, grasp_action=grasp_action), 
                       env, 
                       robot, 
                       grasp_action,
                       episode_memory)

    # Debugging
    post_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - place_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], place_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # breakpoint()
    # ====================================================================================


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
    # move hand up
    current_eef_pose = robot.get_relative_eef_pose(arm='right')
    # current_eef_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
    print("current_eef_pose: ", current_eef_pose)
    noise_x, noise_y, noise_z = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.05)
    up_noise = th.tensor([0.0, 0.0, 0.2]) + th.tensor([noise_x, noise_y, noise_z])
    target_pose = (current_eef_pose[0] + up_noise, current_eef_pose[1])
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action=-1.0)
    
    # move base
    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = -1
    print("action: ", action)
    # base_x_vel = np.random.uniform(-0.01, 0.05)
    base_x_vel = np.random.uniform(0.08, 0.13)
    base_y_vel = np.random.uniform(-0.1, 0.1)
    base_yaw_vel = np.random.uniform(-0.01, 0.01) # 0.2 originally
    # action[:3] = th.tensor([0.0, 0.0, 0.2])
    action[:3] = th.tensor([base_x_vel, base_y_vel, base_yaw_vel])
    env.step(action)
    timesteps = np.random.randint(15, 40)
    for _ in range(timesteps): # was 30 before
        og.sim.step()
    
    # Randomizing head pose
    # default_head_joints = np.array([-0.20317451, -0.7972661])
    # default_head_joints = np.array([-0.5031718015670776, -0.9972541332244873])
    default_head_joints = np.array([-0.503, -0.797])
    noise_1 = np.random.uniform(-0.05, 0.05, 1)
    noise_2 = np.random.uniform(-0.05, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints_pos = default_head_joints + noise
    # head_joints_pos = np.array([0.0, -0.5])
    head_joints_pos = th.from_numpy(head_joints_pos)
    head_joints_pos = th.tensor(head_joints_pos, dtype=th.float32)
    robot.set_joint_positions(head_joints_pos, indices=robot.camera_control_idx)

    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = -1
    env.step(action)

    for _ in range(50):
        og.sim.step()

    set_extrinsic_matrix(robot)

    # add to memory
    dump_to_memory(env, robot, episode_memory)

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
    }
]

env = og.Environment(configs=config)

# og.clear()
# og.sim.restore(["episode_00000_before_place.json"])
# og.sim.restore(["moma_pick_and_place/episode_00000_start.json"])
og.sim.restore(["place_start.json"])

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

# getting all objects in scene
# env.scene.objects

box = env.scene.object_registry("name", "box")
box.root_link.mass = 1e-2
print("box.mass: ", box.mass)

# # debugging
# print("robot left gripper left finger: ", robot.finger_links['left'][0].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())
# print("robot left gripper right finger: ", robot.finger_links['left'][1].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

save_folder = 'temp'
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
for i in range(3):
    print(f"---------------- Episode {i} ------------------")
    episode_memory = Memory()
    
    # randomize base pose and head pose a bit
    randomize_robot()
    # breakpoint()
    
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    primitive(episode_memory)
    episode_memory.dump(f'{save_folder}/dataset.hdf5')
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    
    for _ in range(30):
        og.sim.step()

    og.sim.load_state(state, serialized=False)
    
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


