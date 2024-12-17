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
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.utils.python_utils import nums2array

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
    
    for k in obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k])

    # add relevant object poses
    for obj in relevant_objs:
        for link in obj.links.values():
            pose = link.get_position_orientation()
            pose = T.pose2mat(pose)
            episode_memory.add_value(link.name, pose.cpu().numpy()) 

    is_contact = detect_robot_collision_in_sim(robot)
    
    is_grasping = robot.custom_is_grasping()
    pos_thresh = 0.04
    ori_thresh = 0.1
    reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
    grasp_label = is_grasping and reached_goal
    # grasp_label = is_grasping
    ft_label = reached_goal

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_contact)
    episode_memory.add_extra('reached_goal', reached_goal)
    episode_memory.add_extra('grasp_label', grasp_label)
    episode_memory.add_extra('ft_label', ft_label)


def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    # base_yaw = -10
    base_yaw = np.random.uniform(-20, 20)
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # Randomizing base pos
    # base_pos = np.array([0.55, 0.0, 0.0])
    base_pos = np.array([0.48, 0.0, 0.0])
    base_x_noise = np.random.uniform(-0.2, 0.1)
    base_y_noise = np.random.uniform(-0.1, 0.1)
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

    # # add to memory
    # dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None, check_grasp=False, log=False, writer=None):
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
                # remove the last action from memory
                episode_memory.data['actions']['actions'].pop()
                return

            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            
            # Sidestep control issues. If pose error gets large, means the previous action was bad and so we save that action and stop the episode
            pos_thresh = 0.04
            ori_thresh = 0.1
            reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
            if not reached_goal:
                return obs, info, total_collisions
            
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
        
        # If singularity is reached in this action, stop and do not add to memory
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)
        if sum(singularities) > 3:
            reached_singularity = True
            print("Reached singularity!")
            # remove the last action from memory
            episode_memory.data['actions']['actions'].pop()
            return reached_singularity
        
        # Check grasp
        is_grasping = robot.custom_is_grasping()
        if check_grasp and not is_grasping:
            print("Grasp failed. Exiting.")
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            return

    return obs, info, total_collisions


def grasp_handle(writer=None):
    grasp_action = -1.0

    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # # w.r.t object
    # target_pose_obj = np.array([
    #     [-0.21300013, -0.03880069,  0.96416772,  0.04917458],
    #     [ 0.36973975, -0.11093961, -0.06906067,  0.2582519 ],
    #     [-0.25799225,  0.94084111, -0.05303142,  0.44741508],
    #     [ 0.41167921,  0.15094028,  0.11901726,  0.56770111],
    # ])
    # w.r.t object robotiq gripper
    target_pose_obj = np.array([
        [-0.03007303, -0.98993309, -0.13830437,  0.03215751],
        [-0.16730912, -0.13142948,  0.97710488, -0.3610887 ],
        [-0.98544572,  0.05252409, -0.16167235,  0.30995518],
        [ 0.        ,  0.,          0.,          1.        ],
    ])
    # # w.r.t world robotiq gripper
    # target_pose_world = th.tensor([
    #     [-0.16730904, -0.13142948,  0.9771049 ,  1.10891759],
    #     [ 0.03007308,  0.98993308,  0.13830438, -0.28215751],
    #     [-0.98544573,  0.05252413, -0.16167226,  0.65757006],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ],
    # ])
    # target_pose_world = T.mat2pose(target_pose_world)

    cabinet_pos_world, cabinet_orn_world = scene.object_registry("name", "bottom_cabinet").get_position_orientation()
    cabinet_pose_world = np.eye(4)
    cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    cabinet_pose_world[:3, 3] = cabinet_pos_world
    target_pose_world = cabinet_pose_world @ target_pose_obj
    target_pos_world = target_pose_world[:3, 3]
    target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (th.tensor(target_pos_world, dtype=th.float32), th.tensor(target_orn_world, dtype=th.float32))
    
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose_world, ignore_failure=True, in_world_frame=True), 
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
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # # TODO: Change the indexing here
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # episode_memory.add_action('actions', action_to_add)
    # ==============================================


def primitive(episode_memory=None, episode_number=0, sampled_vector=None, writer=None):    
    grasp_action = 1.0
    print("sampled_vector: ", sampled_vector)
    # ======================= Move hand back ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    # new_pos = curr_pos + th.tensor([-0.3, 0.0, 0.0])
    new_pos = curr_pos + th.tensor(sampled_vector, dtype=th.float32)
    target_pose = (new_pos, curr_orn)
    
    curr_pos_world, curr_orn_world = robot.eef_links["right"].get_position_orientation()
    new_pos_world = curr_pos_world + th.tensor(sampled_vector, dtype=th.float32)
    target_pose_world = (new_pos_world, curr_orn_world)

    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose_world, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory,
                       check_grasp=True,
                       log=True,
                       writer=writer)
    
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory)
    
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose_world[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose_world[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # # =================================================================================

    # # ============= Open grasp =================
    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(40):
    #     og.sim.step()
    # # # save everything to memory
    # # dump_to_memory(env, robot, episode_memory)
    # # # TODO: Change the indexing here
    # # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # # episode_memory.add_action('actions', action_to_add)
    # # # ==========================================

    for _ in range(50):
        og.sim.step()


def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=3)
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
    # visual_only=True,
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
    static_friction=0.5,
    dynamic_friction=0.5,
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
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

for _ in range(20):
    og.sim.step()

save_folder = 'open_drawer_temp2'
os.makedirs(save_folder, exist_ok=True)
episode_memory = Memory()

episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
        episode_number = len(file['data'].keys())
        print("episode_number: ", episode_number)

num_samples = 20
original_vector = np.array([-0.3, 0.0, 0.0])
print("np.linalg.norm(original_vector): ", np.linalg.norm(original_vector))
# sampled_vectors = sample_from_cone(original_vector, max_angle=np.pi/6, num_samples=num_samples, norm_variance=0.4)
sampled_vectors = sample_from_cube(original_vector, num_samples=num_samples)

# show the original vector and the noisy vector in matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')    
ax.quiver(0, 0, 0, original_vector[0], original_vector[1], original_vector[2], color='r')
for sampled_vector in sampled_vectors:
    ax.quiver(0, 0, 0, sampled_vector[0], sampled_vector[1], sampled_vector[2], color='b')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.2, 0.2])
plt.show()


# setting properties of the objects
drawer = env.scene.object_registry("name", "bottom_cabinet")
relevant_objs = [drawer]
drawer.root_link.mass = 50.0
for link_number in range(1, 5):
    drawer.links[f"link_{link_number}"].mass = 5.0


# # remove later
# # ============= Perform grasp ===================
# grasp_action = 1.0
# action = action_primitives._empty_action()
# action[robot.gripper_action_idx["right"]] = grasp_action
# env.step(action)
# for _ in range(40):
#     og.sim.step()
# print("gripper_right_qpos: ", robot._get_proprioception_dict()['gripper_right_qpos'])
# breakpoint()


state = og.sim.dump_state(serialized=False)
for i in range(num_samples):
    custom_reset(env, robot, episode_memory)
    
    imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    output_path = f'{save_folder}/episode_{episode_number:05d}_video.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)
    
    grasp_handle()
    set_extrinsic_matrix(robot)
    # # save the start simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    action_primitives.move_hand_direct_ik_pos_error = 0.0
    action_primitives.move_hand_direct_ik_orn_error = 0.0
    dump_to_memory(env, robot, episode_memory)
    
    primitive(episode_memory, episode_number, sampled_vector=sampled_vectors[i], writer=writer)
    episode_memory.add_action("complete_actions", sampled_vectors[i])
    episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # # save the end simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    # breakpoint()
    og.sim.load_state(state, serialized=False)
    for _ in range(10):
        og.sim.step()

    del episode_memory
    episode_number += 1

    episode_memory = Memory()


breakpoint()

# Always shut down the environment cleanly at the end
og.shutdown()