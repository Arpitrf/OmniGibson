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
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.utils.python_utils import nums2array
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from omnigibson.arpit_trial.utils.data_collection_configs import data_collection_configs as DCC

prior = np.array([
    [0.,    0.,    0., 0.05,  0.00,  0.02,    0.0,    0.,    0.0, -1.0],
    [0.,    0.,    0., 0.05,  -0.00,   0.001, 0.,    0.,    0.0, -1.0],
    [0.,    0.,    0., 0.05,  0.00, 0.02, 0.,    0.,    0.0, -1.0],
    [0.,    0.,    0., 0.05,  -0.00, -0.00, 0.,    0.,    0.0, -1.0],
])

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

def sample_delta_orientation(prior, noise=0.5):
    new_traj = []
    for original_delta_orn in prior:
        original_delta_orn_euler = np.array(R.from_rotvec(original_delta_orn).as_euler("xyz", degrees=False))
        # Decompose the delta orientation into Euler angles
        delta_orn_x, delta_orn_y, delta_orn_z = original_delta_orn_euler  # Assuming (yaw, pitch, roll) order
        
        # Sample noise for each axis
        delta_orn_z = delta_orn_z + np.random.uniform(-1.5*noise, noise) # as hand is pointing inward initially
                
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
        new_traj.append(noisy_vector)

    return np.array(new_traj)

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
    target_pose = (th.tensor(TEP.HELD_POS_LEDGE["robot"], dtype=th.float32), current_eef_pose[1])
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action=grasp_action)

    for _ in range(20):
        og.sim.step()


    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    # pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    # orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

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


def dump_to_memory(env, robot, episode_memory, number_of_collisions=0, reached_singularity=False, reached_joint_limits=False, object_dropped=False):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    gripper_right_tool_link_pose = robot.get_relative_link_pose("gripper_right_tool_link")
    gripper_right_tool_link_pose = th.cat((gripper_right_tool_link_pose[0], gripper_right_tool_link_pose[1]))
    proprio['gripper_right_tool_link_pose'] = gripper_right_tool_link_pose
    arm_right_tool_link_pose = robot.get_relative_link_pose("arm_right_tool_link")
    arm_right_tool_link_pose = th.cat((arm_right_tool_link_pose[0], arm_right_tool_link_pose[1]))
    proprio['arm_right_tool_link_pose'] = arm_right_tool_link_pose

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
    if number_of_collisions > 3:
        is_in_collision = True
    print("is_in_collision: ", number_of_collisions, is_in_collision)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_in_collision)
    # episode_memory.add_extra('singularities', reached_singularity)
    episode_memory.add_extra('joint_limits', reached_joint_limits)
    episode_memory.add_extra('object_dropped', object_dropped)


def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None, check_grasp=False, writer=None, log=False):
    number_of_collisions = 0
    singularities = []
    reached_singularity = False
    reached_joint_limits = False
    for action in ctrl_gen:
        if action == 'Done':
            if log:
                print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(action_primitives.move_hand_direct_ik_orn_error))
            
            reached_joint_limits = False
            normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
            close_to_one = th.isclose(normalized_qpos[:-3], th.tensor(1.0), atol=1e-2)
            close_to_neg_one = th.isclose(normalized_qpos[:-3], th.tensor(-1.0), atol=1e-2)
            any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
            # print("any_close_to_one_or_neg_one: ", any_close_to_one_or_neg_one)
            if any_close_to_one_or_neg_one:
                print("Reached joint limits: ", normalized_qpos)
                reached_joint_limits = True

            # if no collision and goal_reached is False, means bad control. We remove the action from memory and discart this waypoint
            pos_thresh = 0.07
            ori_thresh = 0.25
            reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
            print("reached_goal: ", reached_goal)
            if number_of_collisions < 3 and not reached_goal:
                print("Bad control! Removing action from memory and Exiting.")
                if episode_memory is not None:
                    episode_memory.data['actions']['actions'].pop()
                return False
            
            if episode_memory is not None:      
                dump_to_memory(env, robot, episode_memory, number_of_collisions, reached_singularity=reached_singularity, reached_joint_limits=reached_joint_limits) 
            number_of_collisions = 0
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action[:3], action[14:17])
        obs, reward, terminated, truncated, info = env.step(action)

        # ============================================= Check for collisions =============================================
        # Check if robot right gripper is in collision
        gripper_is_contact = robot.get_gripper_collision(filter_obj=held_obj)
        # print("gripper_collision: ", gripper_is_contact)

        # Check if robot right arm is in collision
        arm_is_contact = False
        # TODO: Remove hardcoding from indices
        for j in range(1,8):
            lis = robot.links[f"arm_right_{j}_link"].contact_list()
            if len(lis) > 0:
                # print(f"arm_right_{j}_link in contact at step {i}: ", lis)
                arm_is_contact = True

        # Check if box is in collision
        held_obj_is_contact = False
        held_obj_contact_bodies = list(held_obj.states[ContactBodies].get_value())
        # don't count gripper fingers as contact
        for contact_body in held_obj_contact_bodies:
            if "robotiq" not in contact_body.name:
                held_obj_is_contact = True
                break  

        # print("arm_is_contact: ", arm_is_contact, "held_obj_is_contact: ", held_obj_is_contact, "gripper_is_contact: ", gripper_is_contact)
        is_contact = arm_is_contact or held_obj_is_contact or gripper_is_contact
        if is_contact:
            number_of_collisions += 1
            # print("Collided! number_of_collisions: ", number_of_collisions)
        # ====================================================================================

        if writer is not None:
            img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
            concat_img = hori_concatenate_image([viewer_img, img])
            concat_img = concat_img * 255.0
            concat_img = concat_img.astype(np.uint8)
            writer.append_data(concat_img)
        
        # if singularity is reached in this episode, do not add to memory
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)

        if sum(singularities) > 10:
            print("Reached singularity!")
            if episode_memory is not None:
                # dump_to_memory(env, robot, episode_memory, reached_singularity=True)
                episode_memory.data['actions']['actions'].pop()
            return False
    
    return True

def primitive(robot, episode_memory=None):
    grasp_action = -1.0
    # ======================= Move hand to place pose ================================
    # w.r.t world
    # place_pose =  (th.tensor([ 1.10402, -0.1873,  0.8563]), th.tensor([-0.0488, -0.0116,  0.5546,  0.8306])) # [ 1.1602, -0.1873,  0.8463]
    # w.r.t robot
    # place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))

    # for vertical
    # w.r.t world
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    place_pose =  (th.tensor([ 1.10402, -0.1873,  0.9563]), curr_orn)
    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory, grasp_action=grasp_action), 
                       env, 
                       robot, 
                       grasp_action,
                       episode_memory)
    # execute_controller(action_primitives._move_hand_direct_ik(place_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action)
    # Debugging
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - place_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], place_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # ====================================================================================

def move_primitive(robot, action_traj, episode_memory=None, writer=None, obj_dropping_episode_momory=None):
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
        action_wrt_robot = np.concatenate((action_wrt_world[:3], homo_action_wrt_robot[:3, 3], np.array(R.from_matrix(homo_action_wrt_robot[:3, :3]).as_rotvec()), action_wrt_world[-1:]))

        if episode_memory is not None:
            episode_memory.add_action('actions', action_wrt_robot)

        current_pose = robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = action_wrt_robot[3:6]
        delta_orn = action_wrt_robot[6:9]
        print("delta_pos, delta_orn: ", delta_pos, delta_orn)
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
                                                                                in_world_frame=False), 
                                                                        env, 
                                                                        robot, 
                                                                        grasp_action, 
                                                                        episode_memory,
                                                                        check_grasp=False,
                                                                        writer=writer,
                                                                        log=True)


        for _ in range(20):
            og.sim.step()

        ee_pose_after = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
        orn_error = orn_error % (2*th.pi)
        # print("prev_pos, target_pos, reached_pos: ", current_pos, target_pos, ee_pose_after[0])
        print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")
        # breakpoint()

        if action_exec is False:
            return 
    
    # ============= Open grasp ===================
    print("Opening grasp")
    robot.controllers["gripper_right"]._use_impedances = True
    for _ in range(20): og.sim.step()
    # breakpoint()
    obj_in_hand_pos_before = held_obj.get_position_orientation()[0]
    object_dropped = False 
    dump_to_memory(env, robot, obj_dropping_episode_momory)

    # breakpoint()
    grasp_action = -1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)

    # # check if the gripper was actullaty opened 
    # gripper_right_qpos = robot._get_proprioception_dict()['gripper_right_qpos']
    # if gripper_right_qpos[0] > 0.2:
    #     print("Gripper did not open. Trying again.")
    #     # breakpoint()
    #     # robot.controllers["gripper_right"]._motor_type = "velocity"
    #     # from omnigibson.controllers.controller_base import ControlType
    #     # robot._joints["right_robotiq_140_joint_finger"].set_control_type(ControlType.POSITION, kp=150.0)
    #     # robot.set_joint_positions(positions=th.tensor([0.02]), indices=robot.gripper_control_idx['right'])
    #     # for _ in range(50): og.sim.step()

    #     grasp_action = -1.0
    #     action = action_primitives._empty_action()
    #     action[robot.gripper_action_idx["right"]] = grasp_action
    #     env.step(action)
    #     for _ in range(40): og.sim.step()
    #     # robot.controllers["gripper_right"]._motor_type = "position"

    robot.controllers["gripper_right"]._use_impedances = False
    for _  in range(50): 
        og.sim.step()
        if writer is not None:
            obs, _ = env.get_obs()
            img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
            concat_img = hori_concatenate_image([viewer_img, img])
            concat_img = concat_img * 255.0
            concat_img = concat_img.astype(np.uint8)
            writer.append_data(concat_img)

    
    action_to_add = np.concatenate((np.zeros(9), np.array([-grasp_action])))
    obj_dropping_episode_momory.add_action('actions', action_to_add)
    
    obj_in_hand_pos_after = held_obj.get_position_orientation()[0]
    delta_pos_z = abs(obj_in_hand_pos_before[2] - obj_in_hand_pos_after[2]) 
    if delta_pos_z > 0.2:
        object_dropped = True
    print("object_dropped, delta_pos_z: ", object_dropped, delta_pos_z)

    dump_to_memory(env, robot, episode_memory=obj_dropping_episode_momory, object_dropped=object_dropped)

    # If the gripper did open, this is a valid data so save it
    gripper_right_qpos = robot._get_proprioception_dict()['gripper_right_qpos']
    print("gripper_right_qpos: ", gripper_right_qpos)
    if gripper_right_qpos[0] < 0.2:    
        print("SUCCESSFUL GRIPPER OPEN. Data saved.")
        obj_dropping_episode_momory.dump(f'{obj_dropping_save_folder}/dataset.hdf5')
    # ==============================================


def randomzie_objects():
    z_scale = np.random.uniform(0.8, 1.4)
    y_scale = np.random.uniform(0.8, 1.0)
    temp_state = og.sim.dump_state(serialized=False)
    og.sim.stop()
    shelf.scale = th.tensor([2.0, 2.0 * y_scale, 1.0 * z_scale])
    og.sim.play()
    og.sim.load_state(temp_state)

    robot.keep_still()
    
    # num_objects = np.random.randint(2, 6)
    num_objects = 6
    chosen_objs = np.random.choice(np.array(extra_objects), num_objects, replace=False)
    chosen_obj_pos = th.tensor([1.3433, -0.150, 0.7241])
    for chosen_obj in chosen_objs:
        chosen_obj = env.scene.object_registry("name", chosen_obj.name)
        # pos_x_noise = np.random.uniform(-0.2, 0.3)
        # pos_y_noise = np.random.uniform(-0.2, 0.4)
        # sampled_pos = chosen_obj_pos + th.tensor([pos_x_noise, pos_y_noise, 0.0])
        # chosen_obj.set_position_orientation(position=sampled_pos)
        chosen_obj.states[object_states.Inside].set_value(other=shelf, new_value=True)

def randomize_robot():
    
    # # Randomize initial hand pose
    # grasp_action=1.0
    # current_eef_pose = robot.get_relative_eef_pose(arm='right') 
    # # Noise range 1 (0-300)
    # noise_x, noise_y, noise_z = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.1)
    # noise = th.tensor([noise_x, noise_y, noise_z])
    # target_pose = (current_eef_pose[0] + noise, current_eef_pose[1])
    # # Ensure the target pose is within the robot's reachable workspace
    # reachable_workspace = robot.arm_reachable_workspace["right"]
    # target_pose = (th.clamp(target_pose[0], min=reachable_workspace["min"], max=reachable_workspace["max"]), target_pose[1])
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
    #                    env, 
    #                    robot, 
    #                    grasp_action=grasp_action)
    # for _ in range(20):
    #     og.sim.step()
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    # pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    # orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    
    # # Randomize initial base pose
    # action = th.zeros(robot.action_dim)
    # action[robot.gripper_action_idx["right"]] = 1
    # # base_x_vel = np.random.uniform(-0.01, 0.05)
    # base_x_vel = np.random.uniform(-0.1, 0.15) # 0.08, 0.13
    # base_y_vel = np.random.uniform(-0.1, 0.1)
    # base_yaw_vel = np.random.uniform(-0.02, 0.02) # 0.2 originally
    # # action[:3] = th.tensor([0.0, 0.0, 0.2])
    # action[:3] = th.tensor([base_x_vel, base_y_vel, base_yaw_vel])
    # env.step(action)
    # timesteps = np.random.randint(15, 40)
    # for _ in range(timesteps):
    #     og.sim.step()
    
    # Randomize head pose
    default_head_joints = np.array([-0.603, -0.797])
    noise_1 = np.random.uniform(-0.2, 0.1, 1)
    noise_2 = np.random.uniform(-0.1, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints_pos = default_head_joints + noise
    head_joints_pos = th.from_numpy(head_joints_pos)
    head_joints_pos = th.tensor(head_joints_pos, dtype=th.float32)
    robot.set_joint_positions(head_joints_pos, indices=robot.camera_control_idx)

    # TODO: Randomize torso height

    # Loop a few physics steps to let the robot settle
    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = 1
    env.step(action)
    for _ in range(20):
        og.sim.step()

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=4)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# set control specific parameters
config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
# config["robots"][0]["controller_config"]["arm_right"]["kp"] = 500.0
# config["robots"][0]["controller_config"]["arm_right"]["kp"] = th.tensor([2000.0, 2000.0, 2000.0, 5000.0, 1000.0, 1000.0, 1000.0])


env = og.Environment(configs=config)

SAVE_VIDEO_FREQUENCY = 10
save_folder = 'place_in_sink'
obj_dropping_save_folder = 'place_in_sink_obj_dropping'
os.makedirs(save_folder, exist_ok=True)
os.makedirs(obj_dropping_save_folder, exist_ok=True)
# Obtain the number of episodes
episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)

og.sim.viewer_camera.set_position_orientation(th.tensor([ 1.1556, -1.0149,  1.5759]), th.tensor([0.5073, 0.0692, 0.1161, 0.8511]))

grasp_modes = ["forward"]
obj_names = ["pan"] 
for obj_name in obj_names:
    for grasp_mode in grasp_modes:
        dcc = DCC[f'place_in_sink_{grasp_mode}_{obj_name}']
        # Set the starting simulation state
        # og.clear()
        og.sim.restore([f"saved_simulation_states/{dcc['start_state']}"])
        # for _ in range(10): og.sim.step()


        scene = env.scene
        robot = env.robots[0]
        breakpoint()
        # robot = og.sim.scenes[0].robots[0]
        action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
        set_gripper_friction()

        # # =========== Adding more objects at runtime ===========
        # # add more objects in the shelf: can, bottle, bowl
        # extra_objects = [
        #     DatasetObject(
        #         name = "can_1",
        #         category = "can_of_baking_mix",
        #         model = "blrqqz", 
        #         scale = [0.6, 0.6, 0.8],
        #         position = [-0.5, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     ),
        #     DatasetObject(
        #         name = "can_2",
        #         category = "can_of_baking_mix",
        #         model = "blrqqz", 
        #         scale = [0.8, 0.8, 1.2],
        #         position = [-0.5, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     ),
        #     DatasetObject(
        #         name = "bowl",
        #         category = "bowl",
        #         model = "tvtive", 
        #         # scale = [0.7, 0.7, 1.0],
        #         position = [-0.7, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     ),
        #     DatasetObject(
        #         name = "box_of_apple_juice_1",
        #         category = "box_of_apple_juice",
        #         model = "zjzgjy", 
        #         # scale = [0.6, 0.6, 0.8],
        #         position = [-0.9, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     ),
        #     DatasetObject(
        #         name = "box_of_apple_juice_2",
        #         category = "box_of_apple_juice",
        #         model = "zjzgjy", 
        #         scale = [0.8, 0.8, 0.8],
        #         position = [-0.9, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     ),
        #     DatasetObject(
        #         name = "box_of_almond_milk",
        #         category = "box_of_almond_milk",
        #         model = "oiiqwq", 
        #         # scale = [0.6, 0.6, 0.8],
        #         position = [-0.9, 0.5, 0.5],
        #         mass = 15.0
        #         # orientation = [0, 0, 0, 1]
        #     )
        # ]
        # for extra_obj in extra_objects:
        #     env.scene.add_object(extra_obj)
        #     extra_obj.root_link.mass = 10.0
        #     random_pos = np.random.uniform(-2.0, -1.0, 2)
        #     extra_obj.set_position_orientation(position=th.tensor([random_pos[0], random_pos[1], 0.0]))
        #     for _ in range(20): og.sim.step()
        #     # breakpoint()
        # # ====================================================

        # ================= Setting object properties =================
        # # env.scene.objects
        # shelf = env.scene.object_registry("name", "shelf")
        # shelf.root_link.mass = 5e4
        # # shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

        held_obj_name = dcc["obj_name"]
        held_obj = env.scene.object_registry("name", held_obj_name)
        held_obj.root_link.mass = 1e-1
        eef_marker = env.scene.object_registry("name", "marker")
        # =============================================================

        # breakpoint()

        # Modify any controller parameters here
        robot.controllers["arm_right"].kp = th.tensor([2000.0, 2000.0, 2000.0, 5000.0, 1000.0, 1000.0, 1000.0])
        # robot.controllers["arm_right"].kp = 500.0
        robot.controllers["gripper_right"].kp = 200.0

        robot.keep_still()
        for _ in range(10): og.sim.step()
        # initialize_robot()

        state = og.sim.dump_state(serialized=False)
        for i in range(dcc["num_episodes"]):
            print(f"---------------- Episode {episode_number} ------------------")
            episode_memory = Memory()
            obj_dropping_episode_momory = Memory()
            writer = None
            
            if i % SAVE_VIDEO_FREQUENCY == 0:
                imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
                output_path = f'{save_folder}/episode_{episode_number:05d}_video.mp4'
                writer = imageio.get_writer(output_path, **imgio_kargs)

            breakpoint()
            # Randomize base pose and head pose a bit
            randomize_robot()
            # randomzie_objects()
            # breakpoint()
            set_extrinsic_matrix(robot)

            # sample trajectory
            # add noise to position
            sampled_traj_pos = sample_from_cone(prior[:, 3:6], max_angle=np.pi/3, norm_variance=0.4)
            sampled_traj = prior.copy()
            sampled_traj[:, 3:6] = sampled_traj_pos
            # add noise to orientation
            delta_orn_euler = R.from_rotvec(sampled_traj[0, 6:9]).as_euler("xyz", degrees=True)
            sampled_traj_orn = sample_delta_orientation(prior[:, 6:9], noise=0.2)
            sampled_traj[:, 6:9] = sampled_traj_orn
            
            action_primitives.move_hand_direct_ik_pos_error = 0.0
            action_primitives.move_hand_direct_ik_orn_error = 0.0
            dump_to_memory(env, robot, episode_memory)

            # primitive(robot, episode_memory)
            move_primitive(robot, sampled_traj, episode_memory=episode_memory, writer=writer, obj_dropping_episode_momory=obj_dropping_episode_momory)    
            # episode_memory.dump(f'{save_folder}/dataset.hdf5')

            for _ in range(30):
                og.sim.step()
                if writer is not None:
                    obs, _ = env.get_obs()
                    img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
                    viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
                    concat_img = hori_concatenate_image([viewer_img, img])
                    concat_img = concat_img * 255.0
                    concat_img = concat_img.astype(np.uint8)
                    writer.append_data(concat_img)
            
            og.sim.load_state(state, serialized=False)
            robot.keep_still()
            for _ in range(10):
                og.sim.step()

            del episode_memory
            del obj_dropping_episode_momory
            episode_number += 1

# Always shut down the environment cleanly at the end
# og.clear()
og.shutdown()


