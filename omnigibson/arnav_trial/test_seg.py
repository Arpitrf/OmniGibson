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
    video_name = "load_dishwasher_test"
    save_image_dir = f"{video_name}_images"
    counter = 0
    fps = 30


    frame_step = 30 // fps

    with open(f"json_files_seg/{video_name}_seg.json", "r") as json_file:
        json_data = json.load(json_file)

    data = np.load(f"/code/omnigibson/arnav_trial/prior_npz_results/{video_name}_prior_results.npz")
    body_trans = data["body_positions"]
    body_orient = data["body_orientations"]
    hand_positions = data["hand_positions"]
    hand_rotations = data["hand_orientations"]

    print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
    print(f"hand_rotations shape: {hand_rotations.shape}")
    print(f"body_trans shape: {body_trans.shape}")
    print(f"body_orient shape: {body_orient.shape}\n\n\n")

    # TRANSFORMATION BY INIT POSE METHOD!!
    body_points = [(t[0], t[2]) for t in body_trans]
    body_yaw = []
    for i in range(body_orient.shape[0]):
        rotmat = R.from_euler("xyz", body_orient[i]).as_matrix()
        unit_vector = np.array([1., 0., 0.])
        direction_vector = np.matmul(rotmat, unit_vector)
        orientation = math.atan2(direction_vector[2], direction_vector[0])
        body_yaw.append(orientation)
        # body_yaw.append(orientation - 3.14/2)
    
    init_matrix = np.array([
        [np.cos(body_yaw[0]), np.sin(body_yaw[0])],
        [-np.sin(body_yaw[0]), np.cos(body_yaw[0])]
    ])

    points_init_frame = []
    yaw_init_frame = []

    for i in range(len(body_points)):
        delta_pos = np.array([body_points[i][0] - body_points[0][0], body_points[i][1] - body_points[0][1]])
        pos_robot_frame = np.matmul(init_matrix, np.transpose(delta_pos))

        points_init_frame.append((pos_robot_frame[0], pos_robot_frame[1]))
        yaw_init_frame.append(body_yaw[i] - body_yaw[0])

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
    
    env = og.Environment(configs=cfg)
    og.sim.enable_viewer_camera_teleoperation()
    controller = StarterSemanticActionPrimitives(env)
    robot = env.robots[0]

    for _ in range(300):
        og.sim.step()

    if save_images:
        os.mkdir(save_image_dir)

    prev_body_pos, prev_body_orn = robot.get_position_orientation()
    print(f"\nStarting Body Position: {prev_body_pos, prev_body_orn}\n")

    prev_hand_pos, prev_hand_orn = robot.eef_links["right"].get_position_orientation()
    prev_hand_pos, prev_hand_orn = controller._get_pose_in_robot_frame([prev_hand_pos, prev_hand_orn])
    print(f"\nStarting Hand Position: {prev_hand_pos, prev_hand_orn}\n")

    for i in range(len(json_data)):
        action = json_data[str(i)]
        clear_debug_drawing()
        if action["action"] == "navigating":
            print("\nNAVIGATION\n")
            start, end = action["range"][0] + 1, action["range"][1] + 1
            for i in range(start, end):
                current_pos, current_yaw = points_init_frame[i], yaw_init_frame[i]
                prev_pos, prev_yaw = points_init_frame[i - 1], yaw_init_frame[i - 1]

                delta_body_pos = np.array([current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]])
                delta_yaw = current_yaw - prev_yaw

                target_body_pos = np.array([prev_body_pos[0] + delta_body_pos[0], prev_body_pos[1] + delta_body_pos[1], prev_body_pos[2]])
                target_body_orn = R.from_quat(prev_body_orn).as_euler("xyz")
                target_body_orn[2] += delta_yaw

                prev_body_pos, prev_body_orn = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()

                actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], target_body_orn[2]))

                for action in actions:
                    if action != "Done":
                        action[5:12] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Left arm and gripper
                        action[12:19] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Right arm and gripper
                        env.step(action)
                if save_images and i != end - 1:
                    color_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()
                    cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
                    counter += 1

            # Hand correction here
            current_pos, current_orn = hand_positions[end - 1], hand_rotations[end - 1]
            prev_pos, prev_orn = hand_positions[start - 1], hand_rotations[start - 1]
            delta_pos = np.subtract(current_pos, prev_pos)
            # delta_pos = np.array([delta_pos[2], delta_pos[0], delta_pos[1]])
            delta_pos = np.array([delta_pos[2] / 2., delta_pos[0] / 2., delta_pos[1] / 2.])
            target_hand_pos = torch.tensor(list(np.add(prev_hand_pos, delta_pos))).to(torch.float32)
            target_hand_orn = torch.tensor(list(R.from_matrix(current_orn).as_quat())).to(torch.float32)

            prev_hand_pos, prev_hand_orn = target_hand_pos, target_hand_orn

            p, current_hand_orientation = robot.eef_links["right"].get_position_orientation()
            _, current_hand_orientataion = controller._get_pose_in_robot_frame([p, current_hand_orientation])
            current_hand_orientation = current_hand_orientation.numpy()

            t_hand_orn = R.from_matrix(hand_rotations[i]).as_quat()
            curr_inv = R.from_quat(current_hand_orientation).inv()
        
            # Multiply the inverse of the first rotation by the second rotation
            delta = (curr_inv * R.from_quat(t_hand_orn)).as_quat()
            result = (R.from_quat(current_hand_orientation) * R.from_quat(delta)).as_quat()

            # actions = controller._move_hand_direct_ik([target_hand_pos, torch.from_numpy(delta).to(torch.float32)], ignore_failure=True, in_world_frame=False)
            actions = controller._move_hand_direct_ik([target_hand_pos, target_hand_orn], ignore_failure=True, in_world_frame=False)

            # Getting debug box
            clear_debug_drawing()
            robot_position, robot_orientation = robot.get_position_orientation()
            robot_to_world = np.eye(4)
            robot_to_world[:3, :3] = R.from_quat(robot_orientation).as_matrix()
            robot_to_world[:3, 3] = np.transpose(robot_position)
            hand_position_robot_frame = np.array([target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], 1])
            hand_position_robot_frame = np.transpose(hand_position_robot_frame)
            target_hand_pos_world_frame = np.matmul(robot_to_world, hand_position_robot_frame)
            extents = 0.05
            while extents > 0.045:
                draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
                            target_hand_pos_world_frame[2]), extents=(extents, extents, extents))
                extents -= 0.001

            steps = 0
            for action in actions:
                if action != "Done":
                    action[5:12] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Left arm and gripper
                    env.step(action)
                    
                    steps += 1
                    if (steps > 300):
                        break
                else:
                    break
            
            if save_images:
                color_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()
                cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
                counter += 1
        
        else:
            print(f"prev_hand_pos: {prev_hand_pos}")
            print("\nMANIPULATION\n")
            start, end = action["range"][0] + 1, action["range"][1] + 1
            start_body_pos = points_init_frame[start - 1]
            start_body_yaw = yaw_init_frame[start - 1]
            for i in range(start, end):
                current_pos, current_orn = hand_positions[i], hand_rotations[i]
                prev_pos, prev_orn = hand_positions[i - 1], hand_rotations[i - 1]

                delta_pos = np.subtract(current_pos, prev_pos)
                # delta_pos = np.array([delta_pos[2], delta_pos[0], delta_pos[1]])
                delta_pos = np.array([delta_pos[2] / 2., delta_pos[0] / 2., delta_pos[1] / 2.])

                target_hand_pos = list(np.add(prev_hand_pos, delta_pos))
                target_hand_orn = list(R.from_matrix(current_orn).as_quat())

                current_body_pos = points_init_frame[i]
                current_body_yaw = yaw_init_frame[i]
                delta_body_pos = np.array([current_body_pos[0] - start_body_pos[0], current_body_pos[1] - start_body_pos[1]])
                delta_body_yaw = current_body_yaw - start_body_yaw
                rot_mat = np.array([
                    [np.cos(delta_body_yaw), np.sin(delta_body_yaw)],
                    [-np.sin(delta_body_yaw), np.cos(delta_body_yaw)]
                ])
                start_body_pos = current_body_pos
                start_body_yaw = current_body_yaw
                target_hand_pos[0] -= delta_body_pos[1]
                target_hand_pos[1] += delta_body_pos[0]
                point = np.array([target_hand_pos[0], target_hand_pos[1]])
                point = np.matmul(rot_mat, np.transpose(point))
                target_hand_pos[0], target_hand_pos[1] = point[0], point[1]
                target_hand_orn = R.from_quat(target_hand_orn).as_euler("xyz")
                target_hand_orn[2] += delta_body_yaw

                prev_hand_pos, prev_hand_orn = target_hand_pos, R.from_euler("xyz", target_hand_orn).as_quat()

                target_hand_pos, target_hand_orn = torch.tensor(target_hand_pos).to(torch.float32), torch.from_numpy(prev_hand_orn).to(torch.float32)

                p, current_hand_orientation = robot.eef_links["right"].get_position_orientation()
                _, current_hand_orientataion = controller._get_pose_in_robot_frame([p, current_hand_orientation])
                current_hand_orientation = current_hand_orientation.numpy()

                t_hand_orn = R.from_matrix(hand_rotations[i]).as_quat()
                curr_inv = R.from_quat(current_hand_orientation).inv()
            
                # Multiply the inverse of the first rotation by the second rotation
                delta = (curr_inv * R.from_quat(t_hand_orn)).as_quat()
                result = (R.from_quat(current_hand_orientation) * R.from_quat(delta)).as_quat()
                
                # actions = controller._move_hand_direct_ik([target_hand_pos, torch.from_numpy(delta).to(torch.float32)], ignore_failure=True, in_world_frame=False)
                actions = controller._move_hand_direct_ik([target_hand_pos, target_hand_orn], ignore_failure=True, in_world_frame=False)

                # Getting debug box
                clear_debug_drawing()
                robot_position, robot_orientation = robot.get_position_orientation()
                robot_to_world = np.eye(4)
                robot_to_world[:3, :3] = R.from_quat(robot_orientation).as_matrix()
                robot_to_world[:3, 3] = np.transpose(robot_position)
                hand_position_robot_frame = np.array([target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], 1])
                hand_position_robot_frame = np.transpose(hand_position_robot_frame)
                target_hand_pos_world_frame = np.matmul(robot_to_world, hand_position_robot_frame)
                extents = 0.05
                while extents > 0.045:
                    draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
                                target_hand_pos_world_frame[2]), extents=(extents, extents, extents))
                    extents -= 0.001

                steps = 0
                for action in actions:
                    if action != "Done":
                        action[5:12] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Left arm and gripper
                        env.step(action)
                        
                        steps += 1
                        if (steps > 300):
                            break
                    else:
                        break

                if save_images:
                    color_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()
                    cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
                    counter += 1
    if save_images:
        output_video = f"result_videos/{video_name}.avi"
        create_video_from_images(image_folder=save_image_dir, output_video=output_video, fps=fps)

    og.shutdown()


#     for i in range (frame_step, hand_positions.shape[0], frame_step):
#         current_pos, current_yaw = points_init_frame[i], yaw_init_frame[i]
#         prev_pos, prev_yaw = points_init_frame[i - frame_step], yaw_init_frame[i - frame_step]

#         delta_body_pos = np.array([current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]])
#         delta_yaw = current_yaw - prev_yaw

#         target_body_pos = np.array([prev_body_pos[0] + delta_body_pos[0], prev_body_pos[1] + delta_body_pos[1], prev_body_pos[2]])

#         target_body_orn = R.from_quat(prev_body_orn).as_euler("xyz")
#         target_body_orn[2] += delta_yaw

#         # current_pos, current_orn = body_trans[i], body_orient[i]
#         # prev_pos, prev_orn = body_trans[i - frame_step], body_orient[i - frame_step]

#         # Getting body deltas
#         # delta_body_pos = np.subtract(current_pos, prev_pos)
#         # delta_body_pos = np.array([delta_body_pos[0], delta_body_pos[2], delta_body_pos[1]]) # Switching the y and z values
#         # target_body_pos = np.add(prev_body_pos, delta_body_pos)

#         # rotmat = R.from_euler("xyz", current_orn).as_matrix()
#         # unit_vector = np.array([1., 0., 0.])
#         # direction_vector = np.matmul(rotmat, unit_vector)
#         # current_orientation = math.atan2(direction_vector[2], direction_vector[0])

#         # rotmat_prev = R.from_euler("xyz", prev_orn).as_matrix()
#         # unit_vector_prev = np.array([1., 0., 0.])
#         # direction_vector_prev = np.matmul(rotmat_prev, unit_vector_prev)
#         # prev_orientation = math.atan2(direction_vector_prev[2], direction_vector_prev[0])

#         # delta_orientation = current_orientation - prev_orientation

#         # target_body_orn = R.from_quat(prev_body_orn).as_euler("xyz")
#         # target_body_orn[2] += delta_orientation

#         prev_body_pos, prev_body_orn = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()

#         # Getting delta arm positions
#         current_pos, current_orn = hand_positions[i], hand_rotations[i]
#         prev_pos, prev_orn = hand_positions[i - frame_step], hand_rotations[i - frame_step]

#         delta_pos = np.subtract(current_pos, prev_pos)
#         delta_pos = np.array([delta_pos[2], delta_pos[0], delta_pos[1]])

#         target_hand_pos = list(np.add(prev_hand_pos, delta_pos))
#         target_hand_orn = list(R.from_matrix(hand_rotations[i]).as_quat())
#         prev_hand_pos, prev_hand_orn = target_hand_pos, target_hand_orn

#         print(f"\nMoving Body {i}")
        
#         actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], target_body_orn[2]))
#         # actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], orientation))

#         steps = 0
#         for action in actions:
#             action[5:12] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Left arm and gripper
#             action[12:19] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Right arm and gripper
#             env.step(action)
        
#         print(f"\nMoving Hand {i}")
#         # actions = controller._move_hand_direct_ik([target_hand_pos, target_hand_orn], ignore_failure=True, in_world_frame=False)

#         # Getting debug box
#         clear_debug_drawing()
#         robot_position, robot_orientation = robot.get_position_orientation()
#         robot_to_world = np.eye(4)
#         robot_to_world[:3, :3] = R.from_quat(robot_orientation).as_matrix()
#         robot_to_world[:3, 3] = np.transpose(robot_position)
#         hand_position_robot_frame = np.array([target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], 1])
#         hand_position_robot_frame = np.transpose(hand_position_robot_frame)
#         target_hand_pos_world_frame = np.matmul(robot_to_world, hand_position_robot_frame)
#         extents = 0.05
#         while extents > 0:
#             draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
#                          target_hand_pos_world_frame[2]), extents=(extents, extents, extents))
#             extents -= 0.001
#         # draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
#         #                  target_hand_pos_world_frame[2]), extents=(0.025, 0.025, 0.025))
#         # draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
#         #                  target_hand_pos_world_frame[2]), extents=(0.0125, 0.0125, 0.0125))

#         # steps = 0
#         # for action in actions:
#         #     action[5:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Left arm and gripper
#         #     env.step(action)
            
#         #     steps += 1
#         #     if (steps > 200):
#         #         break

#         print(f"\n\n\nTarget Pose: {target_hand_pos, target_hand_orn}")
#         actual_pos, actual_orn = robot.eef_links["right"].get_position_orientation()
#         actual_pos, actual_orn = controller._get_pose_in_robot_frame([actual_pos, actual_orn])
#         print(f"Actual Pose: {actual_pos, actual_orn}\n\n")
        
#         if save_images:
#             color_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()
#             cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
#             counter += 1
    
#     if save_images:
#         output_video = f"result_videos/{video_name}.avi"
#         create_video_from_images(image_folder=save_image_dir, output_video=output_video, fps=fps)

if __name__ == "__main__":
    main()