import os
import yaml
import  pdb
import pickle
import cv2
import imageio

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson import object_states
from memory import Memory

def dump_to_memory(env, robot, episode_memory):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
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
    is_contact = detect_robot_collision_in_sim(robot)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_contact)


def first_primitive(primitive_steps_to_perform, episode_memory=None, grasp_mode="vertical"):

    grasp_action = 1.0
    if 1 in primitive_steps_to_perform:
        # ======================= Move base ================================  
        grasp_action = 1.0
        # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
        target_base_pose = th.tensor([0.0, 0.0, 1.57])
        execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory, grasp_action=grasp_action), 
                        env, 
                        robot, 
                        grasp_action, 
                        episode_memory)    
        # for _ in range(50):
        #     og.sim.step()
        curr_base_pos = robot.get_position()
        print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
        # =================================================================================

    if 2 in primitive_steps_to_perform:
        move_to_grasp_pose(grasp_mode, grasp_action)

    if 3 in primitive_steps_to_perform:
        # ============= Perform grasp ===================
        grasp_action = -1.0
        action = action_primitives._empty_action()
        action[robot.gripper_action_idx["right"]] = grasp_action
        env.step(action)
        for _ in range(40):
            og.sim.step()
        if episode_memory is not None:
            # save everything to memory
            dump_to_memory(env, robot, episode_memory)
            # TODO: Change the indexing here
            action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
            episode_memory.add_action('actions', action_to_add)
        # ==============================================
        
    if 4 in primitive_steps_to_perform:
        # ======================= Move hand up ================================  
        curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
        new_pos = curr_pos + th.tensor([0.0, 0.0, 0.2])
        target_pose = (new_pos, curr_orn)
        execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False, episode_memory=episode_memory), 
                        env, 
                        robot, 
                        grasp_action, 
                        episode_memory)
        
        for _ in range(40):
            og.sim.step()
        
        # Debugging
        post_eef_pose = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
        print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
        # =================================================================================

    if 5 in primitive_steps_to_perform:
        # ============= Move base ===================
        # debugging
        ee_pose_before_nav = robot.get_relative_eef_pose(arm='right')
        # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
        target_base_pose = th.tensor([0.456, 0.0257, 0.0]) # [0.526, 0.0257, 0.0]
        execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory, grasp_action=grasp_action),
                        env, 
                        robot, 
                        grasp_action, 
                        episode_memory)    
        for _ in range(50):
            og.sim.step()
        curr_base_pos = robot.get_position_orientation()[0]
        print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
        
        # Debugging
        ee_pose_after_nav = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after_nav[0] - ee_pose_before_nav[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after_nav[1], ee_pose_before_nav[1])
        print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
        # ============================================


temp_prior = th.tensor([
    [ 0.   ,  0.,    -0.301,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.104,  0.334,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.267,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.1,   -0.,    -0.,    -0.,    -1.   ],
    [ 0.   ,  0.,    -1.456,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.459,  0.034,  0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,    -0.109,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.035, -0.016,  0.135, -0.015, -0.103, -0.065, -1.   ],
    # [ 0.   ,  0.,     0.,     0.044, -0.017,  0.142, -0.016, -0.095, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.054, -0.017,  0.102, -0.016, -0.095, -0.067, -1.   ],
    # [ 0.   ,  0.,     0.,     0.038, -0.017,  0.144, -0.016, -0.103, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.078, -0.017,  0.094, -0.016, -0.103, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
])

def perform_grasp(episode_memory=None):
    # ======================= Move hand to grasp pose ================================    
    grasp_action = 1.0
    # w.r.t world
    target_pose = (th.tensor([0.1829, 0.4876, 0.4051]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    # w.r.t robot
    # target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), th.tensor([-0.0256,  0.0228,  0.6444,  0.7640]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                        env, 
                        robot, 
                        grasp_action, 
                        episode_memory) 
    for _ in range(40):
        og.sim.step()
    # current_pose_world = robot.eef_links["right"].get_position_orientation()
    # print("move hand down completed. Desired and Reached right eef pose reached: ", target_pose[0], current_pose_world[0])
    # =================================================================================
        
def move_to_grasp_pose(grasp_mode, grasp_action):
    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    if grasp_mode == "horizontal":
        # horizontal
        # w.r.t robot
        # target_pose:  (th.tensor([ 0.4891, -0.1747,  0.3917]), th.tensor([-0.0224,  0.0234,  0.6525,  0.7571]))
        # w.r.t world
        target_pose = (th.tensor([0.1747, 0.4891, 0.3922]), th.tensor([-3.2351e-02,  6.7136e-04,  9.9674e-01,  7.3904e-02]))
    
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    
    if grasp_mode == "vertical":
        # vertical
        # w.r.t robot
        # target_pose = (th.tensor([ 0.5066, -0.0575,  0.4948]), th.tensor([ 0.4775,  0.5259, -0.5041,  0.4913]))
        # w.r.t world
        target_pose = (th.tensor([0.0933, 0.5011, 0.4953]), th.tensor([ 0.0090, -0.7102,  0.0341, -0.7031]))

    pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                    env, 
                    robot, 
                    grasp_action, 
                    episode_memory) 
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                    env, 
                    robot, 
                    grasp_action, 
                    episode_memory) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================


def get_episode_reward(robot):
    target_pos = [1.1888, -0.1884,  0.8387]
    curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
    dist = np.linalg.norm(target_pos - curr_pos)
    return -dist

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return concatenated_image

def write_moviepy_video(obs_list, name, folder_path, fps=1):
    # print("pathhhhh: ", name)
    obs_list = np.array(obs_list)
    if isinstance(obs_list, np.floating) or obs_list.dtype == "float" or obs_list.dtype == "float32":
        obs_list = (obs_list * 255).astype(np.int16)
    obs_list = [np.int16(element) for element in obs_list]

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(obs_list, fps=fps)
    if not name.endswith(".mp4"):
        name = f"{folder_path}/{name}.mp4"
    clip.write_videofile(f"{name}", fps=fps, logger=None)

def add_noise(temp_prior):
    temp_prior_modified = temp_prior.clone()
    traj_length = len(temp_prior_modified)
    x_noise = np.random.multivariate_normal(mu_x, sigma_x)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z)
    yaw_noise = np.random.multivariate_normal(mu_yaw, sigma_yaw)
    print("yaw_noise: ", yaw_noise)
    # episode noise for x,y,z
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)
    episode_yaw_noise = yaw_noise

    counter = 0
    for i in range(1, traj_length-1):
        if i in [1, 2, 6]:
            # add noise to arm
            noise = th.tensor([x_noise[counter], y_noise[counter], z_noise[counter]])
            temp_prior_modified[i, 3:6] += noise
            counter += 1
        elif i in [3]:
            # add noise to base
            # noise_base_xy = np.random.uniform(-0.1, 0.1, 2)
            noise_base_xy = [0.0, 0.0]
            # noise_base_yaw = np.random.uniform(-0.3, 0.3)
            noise_base_yaw = yaw_noise[0]
            # pdb.set_trace()
            temp_prior_modified[i, 0:3] += th.tensor([noise_base_xy[0], noise_base_xy[1], noise_base_yaw])
    return temp_prior_modified, episode_pos_noise, episode_yaw_noise

def move_primitive(action, episode_memory=None, ik_test=True):
    safe = True

    current_pose = robot.get_relative_eef_pose(arm='right')
    current_pos = current_pose[0]
    current_orn = current_pose[1]
    
    # print("action: ", action)
    delta_pos = action[3:6]
    # print("delta_pos: ", delta_pos)
    delta_orn = action[6:9]
    grasp_action = action[9]
    # print("grasp_action: ", grasp_action)
    
    target_pos = current_pos + delta_pos
    target_pos = target_pos.type(th.FloatTensor)
    # print("current_pos, target_pos: ", current_pos, target_pos)
    # print("type(target_pos): ", type(target_pos))
    target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
    # print("target_orn: ", target_orn, target_orn.as_quat())
    target_orn = th.tensor(target_orn.as_quat())
    target_orn = target_orn.type(th.FloatTensor)

    target_pose = (target_pos, target_orn)
    # print("current_pos, delta_pos: ", current_pos, delta_pos)
    # print("target_pos: ", target_pos)
    # print("current_joint_pos: ", robot.get_joint_positions()[robot.arm_control_idx["right"]])
    # input()
    test_joint_pos = action_primitives._ik_solver_cartesian_to_joint_space(target_pose)
    print("test_joint_pos: ", test_joint_pos)
    if ik_test and test_joint_pos is None:
        safe = False
        return None, None, 0, safe
    
    # action = action_primitives._empty_action()
    # env.step(action, explicit_joints=test_joint_pos)
    # total_collisions1 = 0
    # for _ in range(40):
    #     og.sim.step()

    obs, info, total_collisions1 = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                            stop_on_contact=False,
                                                                            ignore_failure=True,
                                                                            stop_if_stuck=False,
                                                                            in_world_frame=False), 
                                                                    env, 
                                                                    robot, 
                                                                    grasp_action, 
                                                                    episode_memory)
    

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
    # print("current_base_pos, delta_base_pos: ", current_base_pos[:2], delta_base_pos_world[:2])
    # print("current_base_yaw, delta_base_yaw: ", current_base_yaw, delta_base_yaw)
    obs, info, total_collisions2 = execute_controller(action_primitives._navigate_to_pose_direct(target_pose2d), env, 
                       robot, 
                       grasp_action, 
                       episode_memory)


    # Hack to ensure that even if primitive does not return any action (if delta pose is 0), grasp action is performed
    action = action_primitives._empty_action()
    obs, info, total_collisions3 = execute_controller([action], env, 
                       robot, 
                       grasp_action, 
                       episode_memory)

    # print("total_collisions1, total_collisions2, total_collisions3: ", total_collisions1, total_collisions2, total_collisions3)
    total_collisions = max(total_collisions1, total_collisions2, total_collisions3)

    for _ in range(50):
        # print(robot._controllers["arm_right"]._goal)
        og.sim.step()

    ee_pose_after = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
    print(f"----- Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # input()

    return obs, info, total_collisions, safe

# def execute_controller(ctrl_gen, grasp_action):
#     obs, info = env.get_obs()
#     total_collisions = 0
#     for action in ctrl_gen:
#         if action == 'Done':
#             continue
#         action[20] = grasp_action
#         # print("action: ", action)
#         obs, reward, terminated, truncated, info = env.step(action)
#         box = env.scene.object_registry("name", "box")
#         arm_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
#         if arm_in_collision:
#             total_collisions += 1
#         # print("arm_in_collision, total_collisions: ", arm_in_collision, total_collisions)
#     return obs, info, total_collisions

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        img = obs[f"{env.robots[0].name}"][f"{env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
        writer.append_data(img)
        box = env.scene.object_registry("name", "box")
        arm_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
        if arm_in_collision:
            total_collisions += 1
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)
    return obs, info, total_collisions

def correct_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=20.0,
        dynamic_friction=20.0,
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
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat
    
    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

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


set_all_seeds(seed=2)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = f"outputs/run_{current_time}"
os.makedirs(folder_path, exist_ok=True)

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
        # "scale": [0.3, 0.3, 0.3],
        "position": [0, 0.6, 0.3],
        "orientation": [0, 0, 0, 1]
    },
    {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        # "size": 0.05,
        # "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

og.sim.restore(["moma_pick_and_place/episode_00000_start.json"])
# og.sim.restore(["place_start.json"])

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

scene = env.scene
robot = env.robots[0]
correct_gripper_friction()
# custom_reset(env, robot)

# shelf = env.scene.object_registry("name", "shelf")
# coffee_table = env.scene.object_registry("name", "coffee_table")
# shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))
# coffee_table.set_position_orientation(position=th.tensor([10.0, 10.0, 0.0]))

for _ in range(50):
    og.sim.step()


# box = env.scene.object_registry("name", "box")
# coffee_table = env.scene.object_registry("name", "coffee_table")
# shelf = env.scene.object_registry("name", "shelf")


num_samples = 5
num_top_samples = 3
epochs = 10
success = False

mu_x = np.zeros(3)  # Example: 2-dimensional problem
sigma_x = np.eye(3) * 0.003
mu_y = np.zeros(3)  # Example: 2-dimensional problem
sigma_y = np.eye(3) * 0.003
mu_z = np.zeros(3)  # Example: 2-dimensional problem
sigma_z = np.eye(3) * 0.003


def sample_actions(t, actions):
    if t == traj_length:
        print("No action sampling needed.")
        return actions
    
    x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)
    
    actions_org = temp_prior.clone()
    actions_org = actions_org[None, start_idx:-1]
    actions_org = actions_org.repeat(num_samples, 1, 1)
    actions[:, t:, 3:6] = actions_org[:, t:, 3:6] + episode_pos_noise[:, t:]
    return actions

def safe(action, use_hack=False):
    safe = True
    prev_state = og.sim.dump_state()
    box = env.scene.object_registry("name", "box")
    obj_in_hand_pos_before = box.get_position()
    
    _, _, total_collisions, safe = move_primitive(action)

    if use_hack:
        robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
        action = action_primitives._empty_action()
        action[robot.gripper_action_idx["right"]] = 1.0
        env.step(action)
        for _ in range(40):
            # print(robot._controllers["arm_right"]._goal)
            og.sim.step()
            obs, _ = env.get_obs()
            img = obs[f"{env.robots[0].name}"][f"{env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
            writer.append_data(img)

        
        # gripper_pos = robot.get_joint_positions()[robot.gripper_control_idx["right"]]
        # print("gripper finger joint positions after opening: ", gripper_pos)
        # if abs(gripper_pos[0] - 0.045) > 0.01 or abs(gripper_pos[1] - 0.045) > 0.01:
            # input("GRIPPER DID NOT OPEN!!. Press enter to continue")
    
    obj_in_hand_pos_after = box.get_position()
    delta_pos_z = abs(obj_in_hand_pos_before[2] - obj_in_hand_pos_after[2]) 
    # print("delta_pos_z: ", delta_pos_z)
    # print("total_collisions: ", total_collisions)

    normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
    # print("normalized_qpos: ", normalized_qpos)
    close_to_one = th.isclose(normalized_qpos[:-2], th.tensor(1.0), atol=1e-2)
    close_to_neg_one = th.isclose(normalized_qpos[:-2], th.tensor(-1.0), atol=1e-2)
    any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
    if any_close_to_one_or_neg_one:
        safe = False 

    # object dropped (unsafe)
    if delta_pos_z > 0.35:
        safe = False 
    # collisions
    if total_collisions > 0:
        safe = False
    
    if not safe:
        # Hack to make sure that load_state will work. I think there is an issue in using og.sim.load_state() when there are weird collisions
        robot.set_position_orientation(position=th.tensor([-2.0, 0.0, 0.0]))
        og.sim.load_state(prev_state)
        for _ in range(30):
            og.sim.step()
        # input("Reloaded state. Is it ok?")
    
    print("is this action safe? ", safe)
    # input()
    return safe

def check_success():
    box = env.scene.object_registry("name", "box")
    shelf = env.scene.object_registry("name", "shelf")
    obj_in_shelf = box.states[object_states.Inside].get_value(shelf)
    #TODO: Figure out why this doesn't work: robot._ag_obj_in_hand[robot.default_arm]
    grasping = robot.custom_is_grasping()
    success = obj_in_shelf and not grasping
    return success
    # return False

def undo_action(t, action):
    print("Undoing action")
    a = th.cat((-action[t][:-1], action[t][-1:])) 
    move_primitive(a, ik_test=False)


def func(t, actions):
    
    if t == traj_length:
        print("Reached end of recursion")
        # open gripper and see
        a = th.zeros(10)
        a[-1] = 1.0
        # input("open gripper action")
        retval = safe(a, use_hack=True)
        return retval
    
    for action in actions:
        print("--- time step, action: ", t, action[t][3:6])
        ee_pose_before = robot.get_relative_eef_pose(arm='right')
        joint_pos_before = robot.get_joint_positions()[robot.arm_control_idx["right"]]
        sim_state_before = og.sim.dump_state()
        if safe(action[t]):
            # In the current implementation I am performing the action (move_primitive) inside the safe action. This will change later.
            all_failed = func(t+1, actions)

            # if task success
            if check_success():
                print("Task succeeded!")
                all_failed = False 
                return all_failed
            
            # undo the last action. For now try making it go back to exact joint positions
            # action = action_primitives._empty_action()
            # action[robot.gripper_action_idx["right"]] = -1.0
            # env.step(action, explicit_joints=joint_pos_before)
            # for _ in range(40):
            #     og.sim.step()
            undo_action(t, action)        
                    
            ee_pose_after = robot.get_relative_eef_pose(arm='right')
            pos_error = np.linalg.norm(ee_pose_after[0] - ee_pose_before[0])
            orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], ee_pose_before[1])
            # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
            # print("joint_pos_before: ", joint_pos_before)
            joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            # print("joint_pos after rewind: ", joint_pos_after)
            need_reset = any(abs(joint_pos_before - joint_pos_after) > 0.1)
            print("need to call sim.load_state?: ", need_reset)
            if need_reset:
                og.sim.load_state(sim_state_before)
                for _ in range(30):
                    og.sim.step()
                joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                # print("joint_pos after reset: ", joint_pos_after)
            # input("Undid the action. Press enter to continue")

            # reset the shelf in case it has moved
            shelf = env.scene.object_registry("name", "shelf")
            # print("BEFORE self pos, orn: ", shelf.get_position_orientation())
            shelf.set_position_orientation(shelf_pos_orn[0], shelf_pos_orn[1])
            for _ in range(10):
                og.sim.step()
            # print("AFTER self pos, orn: ", shelf.get_position_orientation())

            if all_failed:
                # sample t+1 actions again
                actions = sample_actions(t+1, actions)
                # input("Resampled")

    all_failed = True
    return all_failed 

shelf = env.scene.object_registry("name", "shelf")
shelf_pos_orn = shelf.get_position_orientation()

# for saving videos
current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
base_folder = f"{current_date}"
time_folder = os.path.join(base_folder, current_time)
folder_path = f"outputs_expl/{time_folder}"
os.makedirs(folder_path, exist_ok=True)

imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
output_path = f'{folder_path}/video.mp4'
writer = imageio.get_writer(output_path, **imgio_kargs)

# robot.get_control_dict().get_fcn("eef_right_jacobian_relative")().shape

# robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
# for _ in range(30):
#     og.sim.step()

# # replay nav sub-tasks
# traj_length = 8
# start_idx = 0
# for t in range(start_idx, traj_length):
#     move_primitive(temp_prior[t])
#     if t == 2:
#         perform_grasp()


# try two modes
modes = ["vertical", "horizontal"]

episode_memory = Memory()
primitive_steps_to_perform = np.arange(1, 6)
grasp_mode = "vertical"

for mode in modes:

    # use primitives for the initial actions
    first_primitive(primitive_steps_to_perform, episode_memory=episode_memory, grasp_mode=grasp_mode)
    print("actionsss: ", episode_memory.data["actions"]["actions"])

    traj_length = 3 # for the place subtask
    start_idx = 8

    x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)

    actions = temp_prior.clone()
    actions = actions[None, start_idx:-1]
    actions = actions.repeat(num_samples, 1, 1)
    actions[:, :, 3:6] = actions[:, :, 3:6] + episode_pos_noise
    print("Start actions shape: ", actions.shape)

    # change this
    all_failed = func(t=0, actions=actions)
    # all_failed = True

    if all_failed:
        # rewind to the state just before a grasp
        # 1. rewind nav to place pose
        # TODO: remove the hardcoding
        action = th.tensor(episode_memory.data["actions"]["actions"], dtype=th.float32)
        for t in range(len(action)-1, len(action)-4, -1):
            print("action[t]: ", action[t])
            # input()
            undo_action(t, action)

        # 2. rewind grasp -> move_hand_direct to grasp pose
        move_to_grasp_pose(grasp_mode, grasp_action=-1.0)
        
        # 3. release grasp
        robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
        grasp_action = 1.0
        action = action_primitives._empty_action()
        action[robot.gripper_action_idx["right"]] = grasp_action
        env.step(action)
        for _ in range(40):
            og.sim.step()

        # 4. move_hand_direct to pre-grasp pose and then to the start pose
        curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
        new_pos = curr_pos + th.tensor([0.0, 0.0, 0.1])
        target_pose = (new_pos, curr_orn)
        execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                        env, 
                        robot, 
                        grasp_action, 
                        episode_memory)
        
        target_joint_pos = robot.default_arm_poses[robot.default_arm_pose]
        robot.set_joint_positions(positions=target_joint_pos, indices=robot.arm_control_idx['right'], drive=False)
        for _ in range(60):
            og.sim.step()

        
    
    primitive_steps_to_perform = np.array([2, 3, 4, 5])
    grasp_mode = "horizontal"

# # replay actions
# traj_length = len(temp_prior)
# for t in range(start_idx, traj_length):
#     move_primitive(temp_prior[t])
#     if t == 2:
#         perform_grasp()

for _ in range(500):
    og.sim.step()


og.shutdown()