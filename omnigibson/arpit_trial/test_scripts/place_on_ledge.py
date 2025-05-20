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
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.objects import PrimitiveObject
from omnigibson.utils.python_utils import nums2array

def closest_rotation_matrix(current_rotation, target_rotation):
    """
    Finds the closer rotation matrix (between target and its Z-axis symmetric equivalent) to the current rotation.
    
    Args:
        current_rotation (np.ndarray): Current 3x3 rotation matrix (R_current).
        target_rotation (np.ndarray): Target 3x3 rotation matrix (R_target).
    
    Returns:
        np.ndarray: The closer rotation matrix to the current rotation.
    """
    # Define the 180-degree rotation about the Z-axis
    R_z_180 = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])
    
    # Compute the symmetric equivalent of the target rotation
    flipped_rotation = target_rotation @ R_z_180

    # Compute the Frobenius norm distances
    distance_to_target = np.linalg.norm(current_rotation - target_rotation, ord='fro')
    distance_to_flipped = np.linalg.norm(current_rotation - flipped_rotation, ord='fro')
    
    # Choose the closer rotation matrix
    if distance_to_target <= distance_to_flipped:
        return target_rotation
    else:
        return flipped_rotation


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

def dump_to_memory(env, robot, episode_memory, relevant_objs=[]):
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

    # fig, ax = plt.subplots(2,2)
    # seg_semantic = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_semantic'].cpu().numpy()
    # seg_instance = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance'].cpu().numpy()
    # seg_instance_id = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'].cpu().numpy()
    # rgb = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['rgb'].cpu().numpy()
    # ax[0, 0].imshow(rgb[:, :, :3])
    # ax[0, 1].imshow(seg_semantic)
    # ax[1, 0].imshow(seg_instance)
    # ax[1, 1].imshow(seg_instance_id)
    # plt.show()

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
    
    # base_yaw = 90
    # r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    # r_quat = R.as_quat(r_euler)
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # # randomizing base pos
    # base_pos = np.array([-0.05, -0.4, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # # Reset environment and robot
    # env.reset()
    # robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(th.tensor(action_primitives.move_hand_direct_ik_orn_error)))
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)
    return obs, info, total_collisions

def visualize_axes(target_pose):
    robot_pos, robot_orn = robot.get_position_orientation()
    robot_orn = robot_orn.numpy()
    robot_pose_world = np.eye(4)
    robot_pose_world[:3, :3] = R.from_quat(robot_orn).as_matrix()
    robot_pose_world[:3, 3] = robot_pos
    robot_orn_matrix = R.from_quat(robot_orn).as_matrix()
    eef_pos, eef_orn = robot.eef_links["right"].get_position_orientation()
    eef_pos, eef_orn = eef_pos.numpy(), eef_orn.numpy()
    vector_x, vector_y, vector_z = np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.1])
    # start_x = eef_pos
    # end_x = eef_pos + vector_x
    # start_y = eef_pos
    # end_y = eef_pos + vector_y
    # start_z = eef_pos
    # end_z = eef_pos + vector_z
    # start_x = start_x.tolist()
    # end_x = end_x.tolist()
    # start_y = start_y.tolist()
    # end_y = end_y.tolist()
    # start_z = start_z.tolist()
    # end_z = end_z.tolist()
    # draw_line(start_x, end_x, color=(1.0, 0.0, 0.0, 1.0))
    # draw_line(start_y, end_y, color=(0.0, 1.0, 0.0, 1.0))
    # draw_line(start_z, end_z, color=(0.0, 0.0, 1.0, 1.0))
    # for _ in range(10):
    #     og.sim.step()


    start_pos = robot_pos.numpy() + np.array([0.2, 0.0, 0.2])
    vector_x_base = robot_orn_matrix @ vector_x
    vector_y_base = robot_orn_matrix @ vector_y
    vector_z_base = robot_orn_matrix @ vector_z
    end_x = start_pos + vector_x_base
    end_y = start_pos + vector_y_base
    end_z = start_pos + vector_z_base
    draw_line(start_pos, end_x, color=(1.0, 0.0, 0.0, 1.0))
    draw_line(start_pos, end_y, color=(0.0, 1.0, 0.0, 1.0))
    draw_line(start_pos, end_z, color=(0.0, 0.0, 1.0, 1.0))
    for _ in range(10):
        og.sim.step()

    _ , eef_orn = robot.get_relative_eef_pose(arm='right')
    eef_orn = eef_orn.numpy()
    eef_orn_matrix = R.from_quat(eef_orn).as_matrix()
    vector_x_eef = eef_orn_matrix @ vector_x_base
    vector_y_eef = eef_orn_matrix @ vector_y_base
    vector_z_eef = eef_orn_matrix @ vector_z_base
    end_x = eef_pos + vector_x_eef
    end_y = eef_pos + vector_y_eef
    end_z = eef_pos + vector_z_eef
    draw_line(eef_pos, end_x, color=(1.0, 0.0, 0.0, 1.0))
    draw_line(eef_pos, end_y, color=(0.0, 1.0, 0.0, 1.0))
    draw_line(eef_pos, end_z, color=(0.0, 0.0, 1.0, 1.0))
    for _ in range(10):
        og.sim.step()


    target_pos, target_orn = target_pose
    target_pos, target_orn = target_pos.numpy(), target_orn.numpy()
    target_pos_world = robot_pose_world @ np.array([*target_pos, 1.0])
    target_pos_world = target_pos_world[:3]
    target_orn_matrix = R.from_quat(target_orn).as_matrix()
    vector_x_grasp = target_orn_matrix @ vector_x_base
    vector_y_grasp = target_orn_matrix @ vector_y_base
    vector_z_grasp = target_orn_matrix @ vector_z_base
    end_x = target_pos_world + vector_x_grasp
    end_y = target_pos_world + vector_y_grasp
    end_z = target_pos_world + vector_z_grasp
    draw_line(target_pos_world, end_x, color=(1.0, 0.0, 0.0, 1.0))
    draw_line(target_pos_world, end_y, color=(0.0, 1.0, 0.0, 1.0))
    draw_line(target_pos_world, end_z, color=(0.0, 0.0, 1.0, 1.0))
    for _ in range(10):
        og.sim.step()

    breakpoint()

def primitive(episode_memory=None, episode_number=0):

    # ======================= Move base ================================  
    grasp_action = -1.0
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.0, 0.0, 1.57])
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)    
    # for _ in range(50):
    #     og.sim.step()
    curr_base_pos = robot.get_position()
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    # =================================================================================
    
    # ======================= Move hand to grasp pose ================================    
    in_world_frame = True
    # # horizontal
    # target_pose = th.tensor([
    #     [ 0.09966776,  0.05407733, -0.99355019,  0.14776738],
    #     [ 0.99490699,  0.00968409,  0.10033096,  0.52168679],
    #     [ 0.01504726, -0.99848979, -0.05283672,  0.40618559],
    #     [ 0.        ,  0.,          0.,          0.        ],
    # ]) 
    # # horizontal-forward
    target_pose = th.tensor([
        [ 0.57506982,  0.05240871, -0.81642393,  0.13022119],
        [ 0.81670615,  0.02154281,  0.57665151,  0.46947819],
        [ 0.04780963, -0.99839333, -0.03041389,  0.50455796],
        [ 0.        ,  0.,          0.,          0.        ]
    ])
    # vertical
    # w.r.t robot
    # target_pose = (th.tensor([ 0.4946, -0.1072,  0.4705]), th.tensor([0.0354, 0.9991, 0.0150, 0.0194]))
    # # w.r.t object
    # target_pose = th.tensor([
    #     [-0.00698751, -0.99389619, -0.00377233,  0.06223419],
    #     [-0.81544803,  0.06576812,  0.17366531,  0.34058464],
    #     [ 0.10651827,  0.05080827, -0.89261549,  0.3468574 ],
    #     [-0.36592966, -0.04666542, -0.26758584,  0.31398208],
    # ])
    # # w.r.t world
    # target_pose = th.tensor([
    #     [-0.04365513, -0.998603,   -0.02977036,  0.09332833],
    #     [-0.99833373,  0.04247905,  0.039055,    0.49790302],
    #     [-0.03773583,  0.0314257,  -0.99879349,  0.47100371],
    #     [ 0.        ,  0.,          0.,          0.        ],
    # ])
    
    # # pan
    # target_pose = th.tensor([
    #     [ 0.27262547,  0.83505312, -0.47787198,  0.35828257],
    #     [ 0.31803698,  0.39054868,  0.86390057,  0.49818253],
    #     [ 0.90803515, -0.38750226, -0.15910426,  0.42454916],
    #     [ 0.        ,  0.,          0.,          0.        ],
    # ])
    
    target_pose = T.mat2pose(target_pose)

    pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=in_world_frame), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory) 

    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=in_world_frame), 
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

    # ============= Perform grasp ===================
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(100):
        og.sim.step()
    # ==============================================
        
    # ======================= Move hand up ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')    # save everything to memory

    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.7])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
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

    breakpoint()

    # ============= Move base ===================
    # debugging
    ee_pose_before_nav = robot.get_relative_eef_pose(arm='right')
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.556, 0.0257, 0.0]) # [0.526, 0.0257, 0.0]
    execute_controller(action_primitives._navigate_to_pose_direct(target_base_pose),
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
    
    og.sim.save([f'saved_simulation_states/place_on_ledge_start_state_front.json'])

    # # # ======================= Move hand to place pose ================================
    # # # w.r.t world
    # # # place_pose =  (th.tensor([ 1.10402, -0.1873,  0.8563]), th.tensor([-0.0488, -0.0116,  0.5546,  0.8306])) # [ 1.1602, -0.1873,  0.8463]
    # # # w.r.t robot
    # # # place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))

    # # # for vertical
    # # # w.r.t world
    # # curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    # # place_pose =  (th.tensor([ 1.10402, -0.1873,  0.9563]), curr_orn)
    # # execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory, grasp_action=grasp_action), 
    # #                    env, 
    # #                    robot, 
    # #                    grasp_action,
    # #                    episode_memory)
    # # # execute_controller(action_primitives._move_hand_direct_ik(place_pose, ignore_failure=True, in_world_frame=True), 
    # # #                    env, 
    # # #                    robot, 
    # # #                    grasp_action)
    # # # Debugging
    # # post_eef_pose = robot.eef_links["right"].get_position_orientation()
    # # pos_error = np.linalg.norm(post_eef_pose[0] - place_pose[0])
    # # orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], place_pose[1])
    # # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # # # ====================================================================================

    # # ============= Open grasp =================
    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(40):
    #     og.sim.step()
    # # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # # TODO: Change the indexing here
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    Scene
    # episode_memory.add_action('actions', action_to_add)
    # # ==========================================

    # # # for _ in range(50):
    # # #     og.sim.step()



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "InteractiveTraversableScene"
config["scene"]["scene_model"] = "Benevolence_2_int"
config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "countertop"]
# config["scene"]["load_room_types"]

# config["robots"][0]["controller_config"]["arm_right"]["name"] = "OperationalSpaceController"

config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0

# for forward grasp
box_euler = [0.0, 0.0, -30.0]
# for top-down grasp
# box_euler = [0.0, 0.0, 0.0]
box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
config["objects"] = [
    {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "fqluyq",
        "scale": [1.0, 1.0, 1.3],
        "position": [0, 0.6, 0.3],
        "orientation": [0, 0, 0, 1]
    },
    {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        # "visual_only": True,
        # "size": 0.05,
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    }
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
print(robot.name)

ledge = env.scene.object_registry("name", "countertop_tpuwys_1")
ledge.set_position_orientation(position=th.tensor([1.3047, -0.3750, 0.858]))
robot.set_joint_positions(positions=th.tensor([0.35], dtype=th.float32), indices=[robot.trunk_control_idx])

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(th.tensor([ 1.1556, -1.0149,  1.5759]), th.tensor([0.5073, 0.0692, 0.1161, 0.8511]))


action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
# breakpoint()
custom_reset(env, robot)
set_extrinsic_matrix(robot)
primitive()

for _ in range(500): og.sim.step()

# Always shut down the environment cleanly at the end
og.shutdown()