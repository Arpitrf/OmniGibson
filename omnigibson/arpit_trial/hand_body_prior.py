import os
import time
import json
import yaml
import torch
import math
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils import ui_utils
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import draw_box, clear_debug_drawing, draw_line

from scipy.spatial.transform import Rotation as R
from get_video import create_video_from_images

def main():
    save_images = True
    video_name = "forward_test4"
    save_image_dir = f"{video_name}_images"
    counter = 0
    fps = 5


    frame_step = 30 // fps

    if save_images:
        os.makedirs(save_image_dir, exist_ok=True)

    data = np.load(f"prior_npz_files/{video_name}_prior_results.npz")
    body_trans = data["body_positions"]
    body_orient = data["body_orientations"]
    hand_positions = data["hand_positions"]
    hand_rotations = data["hand_orientations"]

    print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
    print(f"hand_rotations shape: {hand_rotations.shape}")
    print(f"body_trans shape: {body_trans.shape}")
    print(f"body_orient shape: {body_orient.shape}\n\n\n")

    cfg = dict()
    cfg["scene"] = {
        "type" : "Scene",
        "floor_plan_visible" : True
    }
    cfg["robots"] = [
        {
            "type" : "Tiago",
            "name" : "baby_robot",
            "controller_config" : {
                "arm_left" : {
                    "name" : "InverseKinematicsController",
                    "mode" : "pose_delta_ori"
                },
                "arm_right" : {
                    "name" : "InverseKinematicsController",
                    "mode" : "pose_delta_ori"
                }
            },
            "orientation" : R.from_euler("xyz", [0, 0, -(3.14/2 - 3.14/8)]).as_quat()
        }
    ]


    # Pick place apple environment, uncomment this if  using
    # cfg["objects"] = [
    #     {
    #         "type": "DatasetObject",
    #         "name": "table",
    #         "category": "conference_table",
    #         "model": "qzmjrj", 
    #         "position": [(-0.5 + 1) + 0.5, (-0.7 - 1.5) - 0.25, 0.5],
    #         "scale": [1, 1, 0.7],
    #         "orientation": R.from_euler("xyz", [0, 0, -3.14/2]).as_quat()
    #     },
    #     {
    #         "type": "DatasetObject",
    #         "name": "apple", 
    #         "category": "apple",
    #         "model": "omzprq", 
    #         "position": [(-0.5 + 1) + 0.5, (-0.1 - 1.5) - 0.25, 1]
    #     },
    #     {
    #         "type": "DatasetObject",
    #         "name": "plate",
    #         "category": "plate",
    #         "model": "pkkgzc",
    #         "position": [(-0.75 + 1) + 0.5, (-0.7 - 1.5) - 0.25, 1]
    #     }
    # ]
    
    env = og.Environment(configs=cfg)
    og.sim.enable_viewer_camera_teleoperation()
    controller = StarterSemanticActionPrimitives(env)
    controller.arm = "right"
    robot = env.robots[0]

    prev_body_pos, prev_body_orn = robot.get_position_orientation()
    print(f"\n\nStarting Body Position: {prev_body_pos, prev_body_orn}\n\n")

    prev_hand_pos, prev_hand_orn = robot.eef_links["right"].get_position_orientation()
    prev_hand_pos, prev_hand_orn = controller._get_pose_in_robot_frame([prev_hand_pos, prev_hand_orn])
    print(f"\n\nStarting Hand Position: {prev_hand_pos, prev_hand_orn}\n\n")

    for i in range (frame_step, hand_positions.shape[0], frame_step):
        print("Step: ", i)
        current_pos, current_orn = body_trans[i], body_orient[i]
        prev_pos, prev_orn = body_trans[i - frame_step], body_orient[i - frame_step]

        # Getting body deltas
        delta_body_pos = np.subtract(current_pos, prev_pos)
        delta_body_pos = np.array([delta_body_pos[0], delta_body_pos[2], delta_body_pos[1]]) # Switching the y and z values
        target_body_pos = np.add(prev_body_pos, delta_body_pos)

        rotmat = R.from_euler("xyz", current_orn).as_matrix()
        unit_vector = np.array([1., 0., 0.])
        direction_vector = np.matmul(rotmat, unit_vector)
        current_orientation = math.atan2(direction_vector[2], direction_vector[0])

        rotmat_prev = R.from_euler("xyz", prev_orn).as_matrix()
        unit_vector_prev = np.array([1., 0., 0.])
        direction_vector_prev = np.matmul(rotmat_prev, unit_vector_prev)
        print("direction_vector_prev: ", direction_vector_prev)
        prev_orientation = math.atan2(direction_vector_prev[2], direction_vector_prev[0])

        delta_orientation = current_orientation - prev_orientation
        print("curr, prev, delta ori: ", current_orientation, prev_orientation, delta_orientation)

        target_body_orn = R.from_quat(prev_body_orn).as_euler("xyz")
        target_body_orn[2] += delta_orientation


        # Getting delta arm positions
        current_pos, current_orn = hand_positions[i], hand_rotations[i]
        prev_pos, prev_orn = hand_positions[i - frame_step], hand_rotations[i - frame_step]

        delta_pos = np.subtract(current_pos, prev_pos)
        # delta_pos = np.array([delta_pos[0], delta_pos[2], delta_pos[1]])
        delta_pos = np.array([delta_pos[2], delta_pos[0], delta_pos[1]])

        target_hand_pos = list(np.add(prev_hand_pos, delta_pos))
        target_hand_orn = list(R.from_matrix(hand_rotations[i]).as_quat())

        # print(f"\nMoving Body {i}")
        
        # actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], target_body_orn[2]))
        # # actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], orientation))
        # steps = 0
        # for action in actions:
        #     action[5:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Left arm and gripper
        #     action[12:19] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Right arm and gripper
        #     env.step(action)
        
        # Getting debug box
        # clear_debug_drawing()
        # robot_position, robot_orientation = robot.get_position_orientation()
        # Let's say the robot reaches the target body pos and ori (To remove the robot control part from this debugging)
        robot_position, robot_orientation = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()
        
        robot_to_world = np.eye(4)
        robot_to_world[:3, :3] = R.from_quat(robot_orientation).as_matrix()
        robot_to_world[:3, 3] = np.transpose(robot_position)
        hand_position_robot_frame = np.array([target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], 1])
        hand_position_robot_frame = np.transpose(hand_position_robot_frame)
        target_hand_pos_world_frame = np.matmul(robot_to_world, hand_position_robot_frame)
        draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
                         target_hand_pos_world_frame[2]), extents=(0.005, 0.005, 0.005), size=3)
        prev_hand_position_robot_frame = np.array([prev_hand_pos[0], prev_hand_pos[1], prev_hand_pos[2], 1])
        prev_hand_position_robot_frame = np.transpose(prev_hand_position_robot_frame)
        prev_hand_pos_world_frame = np.matmul(robot_to_world, prev_hand_position_robot_frame)
        draw_line(prev_hand_pos_world_frame[:3], target_hand_pos_world_frame[:3], size=3)
        for _ in range(100):
            og.sim.step()

        
        # print(f"\nMoving Hand {i}")
        # actions = controller._move_hand_direct_ik([target_hand_pos, target_hand_orn], ignore_failure=True, in_world_frame=False)
        # steps = 0
        # for action in actions:
        #     action[5:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Left arm and gripper
        #     env.step(action)
            
        #     steps += 1
        #     if (steps > 200):
        #         break

        # print(f"\n\n\nTarget Pose: {target_hand_pos, target_hand_orn}")
        # actual_pos, actual_orn = robot.eef_links["right"].get_position_orientation()
        # actual_pos, actual_orn = controller._get_pose_in_robot_frame([actual_pos, actual_orn])
        # print(f"Actual Pose: {actual_pos, actual_orn}\n\n")
        
        # if save_images:
        #     color_img = og.sim.viewer_camera._get_obs()[0]['rgb']
        #     cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
        #     counter += 1
            
        # change prev to current
        prev_body_pos, prev_body_orn = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()
        prev_hand_pos, prev_hand_orn = target_hand_pos, target_hand_orn

    
    if save_images:
        output_video = f"{video_name}.avi"
        create_video_from_images(image_folder=save_image_dir, output_video=output_video, fps=fps)

    for _ in range(10000):
            og.sim.step()
    
    og.shutdown()

if __name__ == "__main__":
    main()