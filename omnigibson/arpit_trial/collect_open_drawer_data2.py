import os
import yaml
import  pdb
import pickle
import h5py
import imageio
import cv2

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
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory

# TODO: make this w.r.t object
prior = np.array([
    [0.,    0.,    0., -0.055,  0.003,  0.,    0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.074,  0.007, -0.001, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.072,  0.011, -0.004, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.07 ,  0.016, -0.005, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.069,  0.018, -0.002, 0.,    0.,    0., -1.0],
])

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

def visualize_trajectories(actions, start_position, ax=None, color='r'): 
    total_lines = 0
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        prev_position = start_position

        for j in range(trajectory.shape[0]):
            direction = trajectory[j]  # Direction vector at this waypoint
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction
            step = magnitude * direction_normalized

            next_position = prev_position + step
            ax.quiver(prev_position[0], prev_position[1], prev_position[2], step[0], step[1], step[2], color=color)
            # visualize_marker(start_position=prev_position, end_position=next_position, id=total_lines)
            prev_position = prev_position + step  # Move to the new position
            total_lines += 1

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.2, 0.2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')


def move_primitive(action_traj, episode_memory=None, writer=None):
    for action in action_traj:
        if episode_memory is not None:
            episode_memory.add_action('actions', action)

        current_pose = robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = action[3:6]
        delta_orn = action[6:9]
        grasp_action = action[9]
        
        # remove later
        delta_pos = th.tensor([-0.05, -0.02, 0.0])

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
                                                                                in_world_frame=False), 
                                                                        env, 
                                                                        robot, 
                                                                        grasp_action, 
                                                                        episode_memory,
                                                                        check_grasp=True,
                                                                        writer=writer)


        # breakpoint()
        for _ in range(50):
            og.sim.step()

        ee_pose_after = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
        orn_error = orn_error % (2*th.pi)
        print("prev_pos, target_pos, reached_pos: ", current_pos, target_pos, ee_pose_after[0])
        print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")

        # uncomment later
        # if action_exec is False:
        #     return 

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
        new_traj.append(noisy_vector)

    return np.array(new_traj)

def dump_to_memory(env, robot, episode_memory):
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
    # # add gripper+object seg
    # gripper_obj_seg = obtain_gripper_obj_seg(obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'], obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'])
    # episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k])


    is_contact = detect_robot_collision_in_sim(robot)

    is_grasping = robot.custom_is_grasping()
    pos_thresh = 0.02
    ori_thresh = 0.1
    reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
    grasp_label = is_grasping and reached_goal
    ft_label = reached_goal

    episode_memory.add_extra('grasps', is_grasping)
    episode_memory.add_extra('contacts', is_contact)
    # episode_memory.add_extra('reached_goal', reached_goal)
    # episode_memory.add_extra('grasp_label', grasp_label)
    # episode_memory.add_extra('ft_label', ft_label)

    # # debug
    # print("actions: ", np.array(episode_memory.data["actions"]["actions"]).shape)
    # print("rgb: ", np.array(episode_memory.data["observations"]["rgb"]).shape)
    # print("grasps: ", np.array(episode_memory.data["extras"]["grasps"]).shape)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 0
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # randomizing base pos
    base_pos = np.array([0.55, 0.0, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    # add to memory
    # dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None, check_grasp=False, writer=None):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            
            # Hack to sidestep simulation issue (grasp is not lost when moderately bad action)
            pos_thresh = 0.02
            ori_thresh = 0.1
            reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
            if not reached_goal:
                return False

            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)

        if writer is not None:
            img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
            concat_img = hori_concatenate_image([viewer_img, img])
            concat_img = concat_img * 255.0
            concat_img = concat_img.astype(np.uint8)
            writer.append_data(concat_img)

        # Check grasp
        is_grasping = robot.custom_is_grasping()
        # is_grasping2 = robot._ag_obj_in_hand["right"]
        # print("is_grasping, is_grasping2: ", is_grasping, is_grasping2)

        if check_grasp and not is_grasping:
            print("Grasp failed. Exiting.")
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            return False
        
    return True

def grasp_handle(env, robot, episode_memory=None):
    # ======================= Move hand to grasp pose ================================    
    grasp_action = 1.0
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # w.r.t object
    target_pose_obj = np.array([
        [-0.21300013, -0.03880069,  0.96416772,  0.04917458],
        [ 0.36973975, -0.11093961, -0.06906067,  0.2582519 ],
        [-0.25799225,  0.94084111, -0.05303142,  0.44741508],
        [ 0.41167921,  0.15094028,  0.11901726,  0.56770111],
    ])
    cabinet_pos_world, cabinet_orn_world = scene.object_registry("name", "bottom_cabinet").get_position_orientation()
    cabinet_pose_world = np.zeros((4, 4))
    cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    cabinet_pose_world[:3, 3] = cabinet_pos_world
    target_pose_world = cabinet_pose_world @ target_pose_obj
    target_pos_world = target_pose_world[:3, 3]
    target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (th.tensor(target_pos_world), th.tensor(target_orn_world))

    # pre_target_pose = (target_pose_world[0] + th.tensor([0.0, 0.0, 0.1]), target_pose_world[1]) 
    # execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory) 
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose_world, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose_world[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose_world[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

    # ============= Perform grasp ===================
    grasp_action = -1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(100):
        og.sim.step()
    # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # # TODO: Change the indexing here
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # episode_memory.add_action('actions', action_to_add)
    # ==============================================


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

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
# obj_cfg = dict(
#     type="DatasetObject",
#     name="fridge",
#     category="fridge",
#     model="hivvdf",
#     position=[1.5, -0.6, 1.0],
#     # scale=[2.0, 1.0, 1.0],
#     orientation=rot_quat,
#     )
obj_cfg = dict(
    type="DatasetObject",
    name="bottom_cabinet",
    category="bottom_cabinet",
    model="rntwkg",
    position=[1.5, -0.25, 1.0],
    scale=[1.0, 1.0, 1.2],
    orientation=rot_quat,
    )
config["objects"] = [obj_cfg]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)
og.sim.play()
og.sim.load_state(state)

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# pdb.set_trace()
# obj = env.scene.object_registry("name", "box")
# obj.root_link.mass = 1e-2
# print("obj.mass: ", obj.mass)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([0.88,  0.76,  0.98]),
    th.tensor([-0.12,  0.50,  0.83, -0.20]),
)

for _ in range(20):
    og.sim.step()

save_folder = 'open_drawer_new'
os.makedirs(save_folder, exist_ok=True)
episode_memory = Memory()

# Obtain the number of episodes
episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)

num_samples = 300
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

# plt.show()

# original_vector = np.array([-0.3, 0.0, 0.0])
# print("np.linalg.norm(original_vector): ", np.linalg.norm(original_vector))
# # sampled_vectors = sample_from_cone(original_vector, max_angle=np.pi/6, num_samples=num_samples, norm_variance=0.4)
# sampled_vectors = sample_from_cube(original_vector, num_samples=num_samples)

 
# # show the original vector and the noisy vector in matplotlib 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  
# ax.quiver(0, 0, 0, original_vector[0], original_vector[1], original_vector[2], color='r')
# for sampled_vector in sampled_vectors:
#     ax.quiver(0, 0, 0, sampled_vector[0], sampled_vector[1], sampled_vector[2], color='b')
# ax.set_xlim([-0.5, 0.5])
# ax.set_ylim([-0.5, 0.5])
# ax.set_zlim([-0.2, 0.2])
# plt.show()

# for _ in range(100):
#     og.sim.step()

# setting properties of the objects
drawer = env.scene.object_registry("name", "bottom_cabinet")
drawer.root_link.mass = 50.0
drawer.links["link_5"].mass = 5.0
# breakpoint()

# # Shortcut: take to grasp pose already
custom_reset(env, robot, episode_memory)
grasp_handle(env=env, robot=robot, episode_memory=episode_memory)
writer = None

set_extrinsic_matrix(robot)

state = og.sim.dump_state(serialized=False)
for i in range(num_samples):
    print(f"---------------- Episode {episode_number} ------------------")
    # custom_reset(env, robot, episode_memory)
    
    # imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    # output_path = f'{save_folder}/episode_{episode_number:05d}_video.mp4'
    # writer = imageio.get_writer(output_path, **imgio_kargs)

    # remove later
    #  # ============= Perform grasp ===================
    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(100):
    #     og.sim.step()
    # # save everything to memory
    # # dump_to_memory(env, robot, episode_memory)
    # # # TODO: Change the indexing here
    # # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # # episode_memory.add_action('actions', action_to_add)
    # # ==============================================

    # sample trajectory
    sampled_traj_pos = sample_from_cone(prior[:, 3:6], max_angle=np.pi/6, norm_variance=0.4)
    sampled_traj = prior.copy()
    # uncomment later
    # sampled_traj[:, 3:6] = sampled_traj_pos

    # print("sampled_traj_pos: ", sampled_traj_pos)

    # # show the original vector and the noisy vector in matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # visualize_trajectories(sampled_traj[None, ...], np.array([0.0, 0.0, 0.0]), ax=ax)
    # plt.show()

    # add the starting state to memory
    action_primitives.move_hand_direct_ik_pos_error = 0.0
    action_primitives.move_hand_direct_ik_orn_error = 0.0
    dump_to_memory(env, robot, episode_memory)
    
    # primitive(episode_memory, episode_number, sampled_vector=sampled_vectors[i], writer=writer)
    move_primitive(sampled_traj, episode_memory=episode_memory, writer=writer)
    # episode_memory.dump(f'{save_folder}/dataset.hdf5')

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

    breakpoint()
    og.sim.load_state(state, serialized=False)
    
    for _ in range(10):
        og.sim.step()

    del episode_memory
    episode_number += 1

    episode_memory = Memory()


# breakpoint()

# Always shut down the environment cleanly at the end
og.shutdown()