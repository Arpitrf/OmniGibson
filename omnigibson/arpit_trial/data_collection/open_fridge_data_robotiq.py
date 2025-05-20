import os
import yaml
import  pdb
import pickle
import h5py
import imageio
import cv2

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.utils.python_utils import nums2array
from omnigibson.arpit_trial.utils.data_collection_configs import data_collection_configs as DCC

prior = np.array([
    [ 0.,     0.,     0.,    -0.06, 0.040,   0.,     0.,    -0.,     0.0,    -1.   ],
    [ 0.,     0.,     0.,    -0.05, 0.030,  0.,     0.,    -0.,     0.0,    -1.   ],
    [ 0.,     0.,     0.,    -0.05, 0.030,  0.,     0.,    -0.,     0.0,    -1.   ],
    [ 0.,     0.,     0.,    -0.03, 0.020,  0.,     0.,     0.,     0.0,    -1.   ],
    [ 0.,     0.,     0.,    -0.03, 0.020,  0.,     0.,     0.,     0.0,    -1.   ],
])

def set_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=dcc["gripper_friction"],
        dynamic_friction=dcc["gripper_friction"],
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)
    og.sim.play()
    og.sim.load_state(state)

def move_primitive(robot, action_traj, episode_memory=None, writer=None):
    for waypoint_num, action_wrt_world in enumerate(action_traj):

        if waypoint_num > 0:
            # save state
            path = f"{save_folder}/waypoint_{waypoint_num}"
            os.makedirs(path, exist_ok=True)
            num_files = int(len(os.listdir(path)) / 2)
            og.sim.save([f'{path}/state_{num_files:06d}.json'])
            with open(f'{path}/state_{num_files:06d}.pickle', 'wb') as f:
                pickle.dump(og.sim.dump_state(serialized=False), f)

        print("action_wrt_world: ", action_wrt_world[3:6])

        # convert the action from world frame to robot frame
        robot_pose = robot.get_position_orientation()
        robot_pose = T.pose2mat(robot_pose)
        robot_pose[:3, 3] = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
        homo_action_wrt_world = th.eye(4)
        homo_action_wrt_world[:3, :3] = th.tensor(R.from_rotvec(action_wrt_world[6:9]).as_matrix(), dtype=th.float32)
        homo_action_wrt_world[:3, 3] = th.tensor(action_wrt_world[3:6], dtype=th.float32)
        homo_action_wrt_robot = th.linalg.inv(robot_pose) @ homo_action_wrt_world
        # action_wrt_robot = np.concatenate((action_wrt_world[:3], homo_action_wrt_robot[:3, 3], np.array(R.from_matrix(homo_action_wrt_robot[:3, :3]).as_rotvec()), action_wrt_world[-1:]))
        # a quick fix for not transofrming the rotation part of the delta_action
        action_wrt_robot = np.concatenate((action_wrt_world[:3], homo_action_wrt_robot[:3, 3], action_wrt_world[6:9], action_wrt_world[-1:]))


        if episode_memory is not None:
            episode_memory.add_action('actions', action_wrt_robot)

        current_pose = robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = action_wrt_robot[3:6]
        print("delta_pos: ", delta_pos)
        delta_orn = action_wrt_robot[6:9]
        # negating the action here for the robotiq gripper as -1 is open and 1 is close
        grasp_action = -action_wrt_robot[9]
        
        target_pos = current_pos + delta_pos
        target_pos = target_pos.type(th.FloatTensor)
        target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
        target_orn = th.tensor(target_orn.as_quat())
        target_orn = target_orn.type(th.FloatTensor)

        target_pose = (target_pos, target_orn)
        
        action_exec = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                                stop_on_contact=False,
                                                                                ignore_failure=True,
                                                                                stop_if_stuck=False,
                                                                                in_world_frame=False,), 
                                                                        env, 
                                                                        robot, 
                                                                        grasp_action, 
                                                                        episode_memory,
                                                                        check_grasp=True,
                                                                        writer=writer,
                                                                        log=True,
                                                                        pos_thresh=0.06, ori_thresh=0.2)


        for _ in range(50):
            og.sim.step()

        ee_pose_after = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
        orn_error = orn_error % (2*th.pi)
        print("prev_pos, target_pos, reached_pos: ", current_pos, target_pos, ee_pose_after[0])
        print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")

        if action_exec is False:
            return 


def sample_delta_orientation(prior, noise=0.5):
    new_traj = []
    for original_delta_orn in prior:
        original_delta_orn_euler = np.array(R.from_rotvec(original_delta_orn).as_euler("xyz", degrees=False))
        # Decompose the delta orientation into Euler angles
        delta_orn_x, delta_orn_y, delta_orn_z = original_delta_orn_euler  # Assuming (yaw, pitch, roll) order
        
        # Sample noise for each axis
        delta_orn_z = delta_orn_z + np.random.uniform(-noise, noise)
                
        # Return as Euler angles or convert to quaternion/matrix as needed
        new_orientation = R.from_euler('xyz', [delta_orn_x, delta_orn_y, delta_orn_z], degrees=False).as_rotvec()
        new_traj.append(new_orientation)
    return np.array(new_traj)


def sample_from_cone(prior, max_angle=np.pi/18, norm_variance=0.4):
    new_traj = []
    for original_vector in prior:
        original_norm = np.linalg.norm(original_vector)
        original_vector = original_vector / np.linalg.norm(original_vector)  # Normalize input vector

        # Compute the rotation matrix to align [0, 0, 1] with the original vector
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(original_vector, z_axis):
            rotation_matrix = np.eye(3)  # No rotation needed if already aligned
        else:
            rotation_axis = np.cross(z_axis, original_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, original_vector), -1.0, 1.0))
            rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()

        # Sample a random point in the cone aligned with the z-axis
        z = np.cos(max_angle) + (1 - np.cos(max_angle)) * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        random_point = np.array([x, y, z], dtype=np.float64)

        # Apply the rotation to align the point with the original vector
        noisy_vector = rotation_matrix @ random_point

        # Vary the norm of the noisy vector
        varied_norm = original_norm * (1 + np.random.uniform(-norm_variance, norm_variance))
        noisy_vector *= varied_norm

        # print("np.linalg.norm(noisy_vector): ", np.linalg.norm(noisy_vector))
        # set delta_z = 0.0
        noisy_vector[2] = 0.0
        new_traj.append(noisy_vector)

    return np.array(new_traj)

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            # print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return concatenated_image


def sample_from_cube(original_vector, num_samples=10):
    noisy_vectors = []
    for _ in range(num_samples):
        x_random_noise = np.random.uniform(-0.1, 0.1)
        y_random_noise = np.random.uniform(-0.3, 0.3)
        # z_random_noise = np.random.uniform(-0.05, 0.05)
        # x_random_noise = 0.0
        # y_random_noise = -0.2
        z_random_noise = 0.0
        random_noise = np.array([x_random_noise, y_random_noise, z_random_noise])    
        noisy_vector = original_vector + random_noise
        noisy_vectors.append(noisy_vector)

    return np.array(noisy_vectors)

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


def dump_to_memory(env, robot, episode_memory, reached_singularity=False, reached_joint_limits=False):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    proprio['extrinsic_matrix'] = robot._extrinsic_matrix
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    robot_name = env.robots[0].name
    for k in obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k].cpu().numpy())
    
    for k in obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k])

    # # add relevant object poses
    # for obj in relevant_objs:
    #     for link in obj.links.values():
    #         pose = link.get_position_orientation()
    #         pose = T.pose2mat(pose)
    #         episode_memory.add_value(link.name, pose.cpu().numpy()) 

    is_contact = detect_robot_collision_in_sim(robot)
    
    is_grasping = robot.custom_is_grasping()
    pos_thresh = 0.04
    ori_thresh = 0.1
    reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
    grasp_label = is_grasping and reached_goal
    # grasp_label = is_grasping
    ft_label = reached_goal

    # If arm reached a singularity, we do not trust the supervision coming from the simulator as the arm would've gone crazy
    # So, if singularity is reached, we assume the worst has happened
    if reached_singularity:
        is_grasping = False
        is_contact = True
        reached_goal = False
        grasp_label = False
        ft_label = False
    
    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_contact)
    episode_memory.add_extra('reached_goal', reached_goal)
    episode_memory.add_extra('grasp_label', grasp_label)
    episode_memory.add_extra('ft_label', ft_label)

    episode_memory.add_extra('singularities', reached_singularity)
    episode_memory.add_extra('joint_limits', reached_joint_limits)


def custom_reset(env, robot, episode_memory=None, grasp_mode=None): 
    scene_initial_state = env.scene._initial_state
    
    # base_yaw = -42.5
    base_yaw = np.random.uniform(-50, -30)
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # Randomizing base pos
    # base_pos = np.array([0.55, 0.0, 0.0])
    base_pos = np.array([0.71, -0.05, 0.0])
    base_x_noise = np.random.uniform(-0.1, 0.0)
    base_y_noise = np.random.uniform(-0.2, 0.0)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
    print("base_pos: ", base_pos)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.603, -0.897])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None, check_grasp=False, log=False, writer=None, pos_thresh=0.04, ori_thresh=0.1):
    global GLOBAL_TIMESTEP 
    obs, info = env.get_obs()
    total_collisions = 0
    singularities = []
    reached_singularity = False
    for action in ctrl_gen:
        if action == 'Done':
            if log:
                print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(action_primitives.move_hand_direct_ik_orn_error))
            # if joint limits are reached, this action is not a reliable action and so we will skip it
            normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
            # print("normalized_qpos: ", normalized_qpos)
            close_to_one = th.isclose(normalized_qpos[:-3], th.tensor(1.0), atol=1e-2)
            close_to_neg_one = th.isclose(normalized_qpos[:-3], th.tensor(-1.0), atol=1e-2)
            any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
            # print("any_close_to_one_or_neg_one: ", any_close_to_one_or_neg_one)
            if any_close_to_one_or_neg_one:
                if episode_memory is not None:
                    dump_to_memory(env, robot, episode_memory, reached_joint_limits=True)
                    # episode_memory.data['actions']['actions'].pop()
                return False

            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            
            # Sidestep control issues. If pose error gets large, means the previous action was bad and so we save that action and stop the episode
            reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
            if not reached_goal:
                print("Did not reach waypoint. Exiting. Normalized_qpos: ", robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]])
                return False
            
            continue

        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
        

        if writer is not None:
            img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
            concat_img = hori_concatenate_image([viewer_img, img])
            concat_img = concat_img * 255.0
            concat_img = concat_img.astype(np.uint8)
            writer.append_data(concat_img)
        
        # If singularity is reached in this action
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)
        if sum(singularities) > 10:
            print("Reached singularity!")
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory, reached_singularity=reached_singularity)
                # episode_memory.data['actions']['actions'].pop()
            return False
        
        # Check grasp
        is_grasping = robot.custom_is_grasping()
        if check_grasp and not is_grasping:
            print("Grasp failed. Exiting.")
            # breakpoint()
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            return False

    return True


def grasp_handle(grasp_mode):
    grasp_action = -1.0

    # ======================= Move hand to grasp pose ================================  
    # w.r.t object
    # # right_eef_pose_object:  [[-0.896 -0.064 -0.44  -0.255]
    #     [-0.442  0.024  0.897 -0.507]
    #     [-0.047  0.998 -0.05  -0.016]
    #     [ 0.     0.     0.     1.   ]]  
    # w.r.t world 
    target_pose_world = th.tensor([
        [-0.578,  0.015,  0.816,  1.017],
        [-0.815, -0.059, -0.576, -0.875],
        [ 0.04 , -0.998,  0.047,  0.666],
        [ 0.   ,  0.,     0.,     1.   ],
    ])
    target_quat = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (target_pose_world[:3, 3], th.tensor(target_quat, dtype=th.float32))
    

    # cabinet_pos_world, cabinet_orn_world = scene.object_registry("name", "bottom_cabinet").get_position_orientation()
    # cabinet_pose_world = np.eye(4)
    # cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    # cabinet_pose_world[:3, 3] = cabinet_pos_world
    # target_pose_world = cabinet_pose_world @ target_pose_obj
    # target_pos_world = target_pose_world[:3, 3]
    # target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    # target_pose_world = (th.tensor(target_pos_world, dtype=th.float32), th.tensor(target_orn_world, dtype=th.float32))
    
    pre_target_pose = (target_pose_world[0] + th.tensor([-0.04, 0.04, 0.0]), target_pose_world[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action, 
                       pos_thresh=0.06,
                       ori_thresh=0.3,) 
    
    action_exec = execute_controller(action_primitives._move_hand_direct_ik(target_pose_world, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action,
                       pos_thresh=0.06,
                       ori_thresh=0.3,) 
    
    for _ in range(20):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose_world[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose_world[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # if action failed to execute, return
    if not action_exec:
        return action_exec
    # =================================================================================

    # ============= Perform grasp ===================
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # ==============================================

    is_grasping = robot.custom_is_grasping()
    action_exec = is_grasping

    return action_exec


def randomize_objects(fridge):
    z_scale = np.random.uniform(0.9, 1.1)
    temp_state = og.sim.dump_state(serialized=False)
    og.sim.stop()
    fridge.scale = th.tensor([1.0, 1.0, 1.5*z_scale])
    og.sim.play()
    og.sim.load_state(temp_state)
    fridge.keep_still()
    # set initial fridge joint pos as -1
    fridge.joints["j_link_0"].set_pos(-1.0, normalized=True)

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=10)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# robot specific config
config["robots"][0]["default_arm_pose"] = "horizontal"
config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
# config["robots"][0]["controller_config"]["arm_right"]["kp"] = 250.0
config["robots"][0]["controller_config"]["arm_right"]["kp"] = th.tensor([500.0, 500.0, 500.0, 500.0, 250.0, 250.0, 250.0])

# Create and load this object into the simulator
# rot_euler = [0.0, 0.0, -90.0]
# for hivvdf (upside down)
rot_euler = [180.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
obj_cfg = dict(
    type="DatasetObject",
    name="fridge",
    category="fridge",
    model="hivvdf", #petcxr
    position=[1.5, -0.6, 1.0],
    scale=[1.0, 1.0, 1.5],
    orientation=rot_quat,
    )
# obj_cfg = dict(
#     type="DatasetObject",
#     name="bottom_cabinet",
#     category="bottom_cabinet",
#     # visual_only=True,
#     model="bycegi",
#     position=[1.5, -0.25, 1.0],
#     scale=[1.0, 1.0, 1.2],
#     orientation=rot_quat,
#     )
config["objects"] = [obj_cfg]
dcc = DCC[f"open_fridge_{obj_cfg['model']}"]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

set_gripper_friction()

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([0.88,  0.76,  0.98]),
    th.tensor([-0.12,  0.50,  0.83, -0.20]),
)

for _ in range(20):
    og.sim.step()

SAVE_VIDEO_FREQUENCY = 15
save_folder = 'open_fridge_temp'
os.makedirs(save_folder, exist_ok=True)
episode_memory = Memory()
episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
        episode_number = len(file['data'].keys())
        print("episode_number: ", episode_number)
init_episode_number = episode_number

# # To quickly visualize different trajectories
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  
# visualize_trajectories(prior[None, ...], np.array([0.0, 0.0, 0.0]), ax=ax, color="g")
# for j in range(20):
#     sampled_traj_pos = sample_from_cone(prior[:, 3:6], max_angle=np.pi/6, norm_variance=0.4)
#     sampled_traj = prior.copy()
#     sampled_traj[:, 3:6] = sampled_traj_pos
#     # show the original vector and the noisy vector in matplotlib
#     visualize_trajectories(sampled_traj[None, ...], np.array([0.0, 0.0, 0.0]), ax=ax)


# setting properties of the objects
object_name = "fridge"
fridge = env.scene.object_registry("name", object_name)
fridge.root_link.mass = 50.0
fridge.links["link_0"].mass = 20.0
fridge.joints["j_link_0"].friction = 2.0
fridge.keep_still()

# robot controller settings
# robot.controllers["arm_right"].kp = 300.0

state = og.sim.dump_state(serialized=False)
grasp_modes = dcc["grasp_modes"]

for grasp_mode in grasp_modes:
    while episode_number < init_episode_number + dcc["num_episodes"]:
        print(f"============== Episode {episode_number} ==============")
        writer = None
        
        custom_reset(env, robot, episode_memory, grasp_mode=grasp_mode)
        randomize_objects(fridge)
        robot.keep_still()
        
        action_exec = grasp_handle(grasp_mode)
        print("action_exec: ", action_exec)
        if action_exec is False:
            continue

        if episode_number % SAVE_VIDEO_FREQUENCY == 0:
            imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
            output_path = f'{save_folder}/episode_{episode_number:05d}_video.mp4'
            writer = imageio.get_writer(output_path, **imgio_kargs)

        set_extrinsic_matrix(robot)

        # sample trajectory
        sampled_traj_pos = sample_from_cone(prior[:, 3:6], max_angle=np.pi/3, norm_variance=0.4)
        sampled_traj = prior.copy()
        sampled_traj[:, 3:6] = sampled_traj_pos
        # add noise to orientation
        delta_orn_euler = R.from_rotvec(sampled_traj[0, 6:9]).as_euler("xyz", degrees=True)
        sampled_traj_orn = sample_delta_orientation(prior[:, 6:9], noise=0.15)
        sampled_traj[:, 6:9] = sampled_traj_orn
        
        action_primitives.move_hand_direct_ik_pos_error = 0.0
        action_primitives.move_hand_direct_ik_orn_error = 0.0
        dump_to_memory(env, robot, episode_memory)
        
        # primitive(episode_memory, episode_number, sampled_vector=sampled_vectors[i], writer=writer)
        move_primitive(robot, sampled_traj, episode_memory=episode_memory, writer=writer)
        episode_memory.dump(f'{save_folder}/dataset.hdf5')

        for _ in range(10):
            og.sim.step()
            if writer is not None:
                obs, _ = env.get_obs()
                img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
                viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
                concat_img = hori_concatenate_image([viewer_img, img])
                concat_img = concat_img * 255.0
                concat_img = concat_img.astype(np.uint8)
                writer.append_data(concat_img)

        # breakpoint()
        og.sim.load_state(state, serialized=False)
        for _ in range(10):
            og.sim.step()

        del episode_memory
        episode_number += 1

        episode_memory = Memory()

# Always shut down the environment cleanly at the end
og.shutdown()