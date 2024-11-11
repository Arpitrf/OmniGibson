import cv2
import numpy as np
import torch as th

import omnigibson as og

from omnigibson import object_states
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

def correct_gripper_friction(robot):
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=4.0,
        dynamic_friction=4.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def check_success(env, robot):
    box = env.scene.object_registry("name", "box")
    shelf = env.scene.object_registry("name", "shelf")
    obj_in_shelf = box.states[object_states.Inside].get_value(shelf)
    #TODO: Figure out why this doesn't work: robot._ag_obj_in_hand[robot.default_arm]
    grasping = robot.custom_is_grasping()
    success = obj_in_shelf and not grasping
    return success
    # return False

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

# def add_noise(temp_prior):
#     temp_prior_modified = temp_prior.clone()
#     traj_length = len(temp_prior_modified)
#     x_noise = np.random.multivariate_normal(mu_x, sigma_x)
#     y_noise = np.random.multivariate_normal(mu_y, sigma_y)
#     z_noise = np.random.multivariate_normal(mu_z, sigma_z)
#     yaw_noise = np.random.multivariate_normal(mu_yaw, sigma_yaw)
#     print("yaw_noise: ", yaw_noise)
#     # episode noise for x,y,z
#     episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
#                     np.expand_dims(y_noise, axis=1), 
#                     np.expand_dims(z_noise, axis=1)), axis=1)
#     episode_yaw_noise = yaw_noise

#     counter = 0
#     for i in range(1, traj_length-1):
#         if i in [1, 2, 6]:
#             # add noise to arm
#             noise = th.tensor([x_noise[counter], y_noise[counter], z_noise[counter]])
#             temp_prior_modified[i, 3:6] += noise
#             counter += 1
#         elif i in [3]:
#             # add noise to base
#             # noise_base_xy = np.random.uniform(-0.1, 0.1, 2)
#             noise_base_xy = [0.0, 0.0]
#             # noise_base_yaw = np.random.uniform(-0.3, 0.3)
#             noise_base_yaw = yaw_noise[0]
#             # pdb.set_trace()
#             temp_prior_modified[i, 0:3] += th.tensor([noise_base_xy[0], noise_base_xy[1], noise_base_yaw])
#     return temp_prior_modified, episode_pos_noise, episode_yaw_noise