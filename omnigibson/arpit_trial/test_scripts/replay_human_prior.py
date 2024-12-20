import os
import time
import json
import yaml
import math
import cv2
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch as th
th.set_printoptions(precision=3, sci_mode=False)

from torchvision import transforms
from PIL import Image
from pathlib import Path

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.utils import ui_utils
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import draw_box, clear_debug_drawing, draw_line

from scipy.spatial.transform import Rotation as R

def execute_controller(ctrl_gen, env, robot, grasp_action=-1.0):
    for action in ctrl_gen:
        if action == 'Done':
            # print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(th.tensor(action_primitives.move_hand_direct_ik_orn_error)))
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

def custom_reset(env, robot):
    scene_initial_state = env.scene._initial_state
    
    base_yaw = -90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # # randomizing base pos
    # base_pos = np.array([-0.05, -0.4, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # # set head joint positions
    # head_joints = th.tensor([-0.503, -0.997])
    # robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def main():

    set_all_seeds(seed=0)
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"] = dict()
    config["scene"]["type"] = "Scene"
    
    env = og.Environment(configs=config)
    config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
    config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0
    scene = env.scene
    robot = env.robots[0]
    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    save_images = False
    video_name = "open_fridge1"
    save_image_dir = f"/code/omnigibson/arnav_trial/{video_name}_images"
    counter = 0
    fps = 10
    frame_step = 30 // fps

    if save_images:
        os.mkdir(save_image_dir)

    data = np.load(f"/home/arpit/projects/slahmr-hamer/slahmr/prior_npz_results/{video_name}_prior_results.npz")
    body_trans = data["body_positions"]
    body_orient = data["body_orientations"]
    hand_positions = data["hand_positions"]
    hand_rotations = data["hand_orientations"]

    print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
    print(f"hand_rotations shape: {hand_rotations.shape}")
    print(f"body_trans shape: {body_trans.shape}")
    print(f"body_orient shape: {body_orient.shape}\n\n\n")

    # custom_reset(env, robot)

    # --------------------------------------- WORKING -------------------------------------------
    points = [(t[0], t[2]) for t in body_trans]
    root_orient = data["body_orientations"]

    # Getting orientations
    yaw = []
    for i in range(root_orient.shape[0]):
        rotmat = R.from_euler("xyz", root_orient[i]).as_matrix()
        unit_vector = np.array([1., 0., 0.])
        direction_vector = np.matmul(rotmat, unit_vector)
        orientation = math.atan2(direction_vector[2], direction_vector[0])
        # yaw.append(orientation)
        yaw.append(orientation - 3.14/2)
    
    breakpoint()
    init_matrix = np.array([
        [np.cos(yaw[0]), np.sin(yaw[0])],
        [-np.sin(yaw[0]), np.cos(yaw[0])]
    ])

    delta_points_init_robot_frame = []
    delta_yaws_init_robot_frame = []

    for i in range(len(points)):
        delta_pos = np.array([points[i][0] - points[0][0], points[i][1] - points[0][1]])
        delta_pos_init_robot_frame = np.matmul(init_matrix, np.transpose(delta_pos))
        # print("pos_robot_frame: ", pos_robot_frame)

        delta_points_init_robot_frame.append([delta_pos_init_robot_frame[0], delta_pos_init_robot_frame[1], 0.0])
        delta_yaws_init_robot_frame.append(yaw[i] - yaw[0])
        print("yaw[i] - yaw[0]: ", yaw[i] - yaw[0])


    delta_points_init_robot_frame = np.array(delta_points_init_robot_frame)
    delta_yaws_init_robot_frame = np.array(delta_yaws_init_robot_frame)
    print("delta_points_init_robot_frame: ", delta_points_init_robot_frame.shape)
    print("delta_yaws_init_robot_frame: ", delta_yaws_init_robot_frame.shape)

    T_R0_to_world = robot.get_position_orientation()
    T_R0_to_world = T.pose2mat(T_R0_to_world)
    R_R0_to_world = T_R0_to_world[:3, :3]

    # to segment out specific frames 
    step_size = 5
    start = 65
    end = 90
    # for i in range(step_size, len(delta_points_init_robot_frame), step_size):
    for i in range(start, end, step_size):
        print("======== step: ", i)
        # 1) moving base
        # convert base delta poses from init robot frame to world frame. Note that the delta_pose is (current_pose - init_pose) and not (current_pose - prev_pose)
        delta_pos_world_frame = R_R0_to_world @ np.transpose(delta_points_init_robot_frame[i])
        init_pos =  T_R0_to_world[:3, 3]
        target_pos_world_frame = init_pos + delta_pos_world_frame
        init_yaw = R.from_matrix(R_R0_to_world).as_euler("xyz")[2]
        # since yaw is along z axis which is the same for init robot frame and world frame, delta_yaw_init_robot_frame = delta_yaw_world_frame
        target_yaw_world_frame = init_yaw + delta_yaws_init_robot_frame[i]
        pose2d = (target_pos_world_frame[0], target_pos_world_frame[1], target_yaw_world_frame)
        print("delta_points_init_robot_frame[i]: ", delta_points_init_robot_frame[i])
        print("delta_yaws_init_robot_frame[i]: ", delta_yaws_init_robot_frame[i])
        execute_controller(action_primitives._navigate_to_pose_direct(pose2d), env, robot)

        for _ in range(10):
            og.sim.step()

        # 2) moving hand
        current_human_hand_pos, _ = hand_positions[i], hand_rotations[i]
        prev_human_hand_pos, _ = hand_positions[i - frame_step], hand_rotations[i - frame_step]
        delta_human_hand_pos = current_human_hand_pos - prev_human_hand_pos
        delta_human_hand_pos = np.array([delta_human_hand_pos[2], delta_human_hand_pos[0], delta_human_hand_pos[1]])
        current_robot_hand_pos, current_robot_hand_quat = robot.get_relative_eef_pose(arm='right')
        target_robot_hand_pos = current_robot_hand_pos + delta_human_hand_pos
        target_robot_hand_orn = current_robot_hand_quat
        print("delta_human_hand_pos: ", delta_human_hand_pos)
        target_pose = (target_robot_hand_pos, target_robot_hand_orn)
        breakpoint()
        execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), env, robot)


    og.shutdown()

if __name__ == "__main__":
    main()