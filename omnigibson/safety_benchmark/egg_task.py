import os
import time
import json
import yaml
import math
import cv2
import shutil
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch as th
th.set_printoptions(precision=3, sci_mode=False)

from torchvision import transforms
from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson.utils import ui_utils
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import draw_box, clear_debug_drawing, draw_line, KeyboardRobotController
from omnigibson.arnav_trial.get_video import create_video_from_images

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

def main(config_filename, task_name, save_images=True, save_dir="omnigibson/safety_benchmark/images"):
    set_all_seeds(seed=0)
    config = yaml.load(open(f"omnigibson/safety_benchmark/envs/{config_filename}", "r"), Loader=yaml.FullLoader)
    config["scene"] = dict()
    config["scene"]["type"] = "Scene"

    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "conference_table",
            "model": "qzmjrj", 
            "position": [0.75, 0., 0.],
            "scale": [1, 1, 0.7],
            "orientation": R.from_euler("xyz", [0, 0, -3.14/2]).as_quat()
        },
        {
            "type": "DatasetObject",
            "name": "egg", 
            "category": "egg",
            "model": "brkitw", 
            "position": [0.5, -0.25, 1],
            "scale": [1.5, 1.5, 1.5]
        },
        {
            "type": "DatasetObject",
            "name": "plate",
            "category": "plate",
            "model": "pkkgzc",
            "position": [0.6, 0.1, 1]
        }
    ]
    
    env = og.Environment(configs=config)
    og.sim.enable_viewer_camera_teleoperation()
    config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
    config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0
    scene = env.scene
    robot = env.robots[0]
    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=True)

    head_joints = th.tensor([-0.103, -0.897])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    for _ in range(100):
        og.sim.step()
    
    if save_images:
        obs, _ = env.get_obs()
        color_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()
        robot_img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
        robot_img = cv2.resize(cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB), (512, 512))
        cv2.imwrite(os.path.join(save_dir, f"{task_name}.jpg"), cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_dir, f"{task_name}_robot.jpg"), robot_img)

    action_generator = KeyboardRobotController(robot=robot)
    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()
    breakpoint()
    obs, _ = env.get_obs()
    robot_img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
    robot_img = cv2.resize(cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB), (512, 512))
    cv2.imwrite(os.path.join(save_dir, f"{task_name}_robot_current_img.jpg"), robot_img)
    breakpoint()
    max_steps = -1 
    step = 0
    gripper_close = False
    while step != max_steps:
        action, keypress_str = action_generator.get_teleop_action()
        print(keypress_str, action)
        if keypress_str == 'T':
            gripper_close = not gripper_close
        if gripper_close:
            action[robot.gripper_action_idx["right"]] = 1.0
        else:
            action[robot.gripper_action_idx["right"]] = -1.0
        # print("action: ", action)
        env.step(action=action)
        obs, _ = env.get_obs()
        robot_img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
        robot_img = cv2.resize(cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB), (512, 512))
        # cv2.imwrite(os.path.join(save_dir, f"{task_name}_robot_current_img.jpg"), robot_img)
        if keypress_str == 'TAB':
            break
            # right_eef_pos_world, right_eef_orn_world = robot.eef_links["right"].get_position_orientation()
            # right_eef_pose_world = np.eye(4)
            # right_eef_pose_world[:3, :3] = R.from_quat(right_eef_orn_world).as_matrix()
            # right_eef_pose_world[:3, 3] = right_eef_pos_world

            # obj_pos_world, obj_orn_world = scene.object_registry("name", object_name).get_position_orientation()
            # obj_pose_world = np.eye(4)
            # obj_pose_world[:3, :3] = R.from_quat(obj_orn_world).as_matrix()
            # obj_pose_world[:3, 3] = obj_pos_world

            # if np.linalg.det(obj_pose_world) != 0:
            #     right_eef_pose_object = np.linalg.inv(obj_pose_world) @ right_eef_pose_world
            # else:
            #     right_eef_pose_object = np.linalg.pinv(obj_pose_world) @ right_eef_pose_world

            # base_pose = robot.get_position_orientation()
            # right_eef_pose = robot.get_relative_eef_pose(arm='right')
            # print("right_eef_pose: ", right_eef_pose)
            # print("right_eef_pose_world: ", right_eef_pose_world)
            # print("right_eef_pose_object: ", right_eef_pose_object)
            # print("base_pose: ", base_pose)
            breakpoint()

        step += 1


    og.shutdown()


if __name__ == "__main__":
    config_filename = "egg_task.yaml"
    task_name = "pick_place_egg"
    main(config_filename, task_name)