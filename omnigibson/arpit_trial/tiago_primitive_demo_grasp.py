import math
import os
import pickle
import imageio
import cv2
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects import PrimitiveObject
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm, macros
from omnigibson.object_states import Touching
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson import object_states

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

def save_to_video(obs, episode_error=None):
    img = obs[f"{env.robots[0].name}"][f"{env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255
    viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255
    concat_img = hori_concatenate_image([viewer_img, img]) 
    concat_img = concat_img * 255
    concat_img = concat_img.astype(np.uint8)
    cv2.putText(concat_img, f"Episode {i:03d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # make this text go lower
    if episode_error is not None:
        cv2.putText(concat_img, f"Reason: {episode_error['reason']}. Phase: {episode_error['phase']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    writer.append_data(concat_img)

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        obs, reward, terminated, truncated, info = env.step(action)
        save_to_video(obs)

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=4)
cfg = {
    "env": {
        "action_frequency": 30,
        "physics_frequency": 300,
    },
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors", "breakfast_table", "walls"],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "cologne",
            "category": "bottle_of_cologne",
            "model": "lyipur",
            "position": [1.5629, 0.20, 0.9], #[1.1629, 0.0040, 0.9],
            "orientation": [0, 0, 0, 1],
        },
    ],
    "robots": [
        {
            # "type": "R1",
            "type": "Tiago",
            "obs_modalities": "rgb",
            "position": [0, 0, 0],
            "orientation": [0, 0, 0, 1],
            "self_collisions": True,
            "action_normalize": False,
            "rigid_trunk": False,
            "grasping_mode": "sticky",
            "default_trunk_offset": 0.35,
            "default_arm_pose": "vertical",
            "controller_config": {
                "base": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 1000.0,
                    "pos_ki": 0.5,
                    "max_integral_error": 5.0,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 1000.0,
                    "pos_ki": 0.5,
                    "max_integral_error": 5.0,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
            },
        }
    ],
}

# for saving videos
current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
base_folder = f"{current_date}"
time_folder = os.path.join(base_folder, current_time)
folder_path = f"outputs_data_gen/{time_folder}"
os.makedirs(folder_path, exist_ok=True)
imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}

env = og.Environment(configs=cfg)
robot = env.robots[0]

marker = PrimitiveObject(
    relative_prim_path=f"/marker",
    name="marker",
    primitive_type="Cone",
    scale=0.05,
    # radius=0.03,
    visual_only=True,
    rgba=[1.0, 0, 1.0, 0.5],
    position=[100.0, 100.0, 0.0],
    orientation=[0, 0, 0, 1],
)
env.scene.add_object(marker)

# og.sim.viewer_camera.set_position_orientation([-1.6840, 1.2508, 1.5873], [-0.2597, 0.5574, 0.7147, -0.3332])
og.sim.viewer_camera.set_position_orientation([0.2723, -2.126, 1.8784], [0.538, -0.107, -0.163, 0.819])

# Open the gripper(s) to match cuRobo's default state
for arm_name in robot.gripper_control_idx.keys():
    grpiper_control_idx = robot.gripper_control_idx[arm_name]
    robot.set_joint_positions(th.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
robot.keep_still()

for _ in range(5):
    og.sim.step()

env.scene.update_initial_state()
env.scene.reset()


action_primitives = StarterSemanticActionPrimitives(robot, enable_head_tracking=True, debug_visual_marker=marker)

breakfast_table = env.scene.object_registry("name", "breakfast_table_skczfi_0")
og.sim.stop()
breakfast_table.scale += th.tensor([0.0, 0.0, -0.4])
og.sim.play()
init_state = og.sim.dump_state()
    
episode_errors = {}
for i in range(50):
    print(f"========== Episode {i} ==========")
    output_path = f'{folder_path}/episode_{i:03d}.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)

    cologne = env.scene.object_registry("name", "cologne")
    cologne.states[object_states.OnTop].set_value(breakfast_table, True)
    for _ in range(20):
        og.sim.step()
        obs, info = env.get_obs()
        save_to_video(obs)
    
    print("Executing controller")
    execute_controller(action_primitives.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, cologne), env)
    
    # breakpoint()
    # TODO: change this to entire action primitive object
    if len(action_primitives.errors) > 0:
        episode_errors[i] = dict()
        episode_errors[i]["reason"] = action_primitives.errors[0].reason.name
        episode_errors[i]["phase"] = action_primitives.phase
        episode_errors[i]["metadata"] = action_primitives.errors[0].metadata
    else:
        episode_errors[i] = dict()
        episode_errors[i]["reason"] = "None"
        episode_errors[i]["phase"] = "None"
        episode_errors[i]["metadata"] = "None"
    print("action_primitives.errors", action_primitives.errors)
    print("Finished executing grasp")

    for _ in range(50):
        og.sim.step()
        obs, info = env.get_obs()
        save_to_video(obs, episode_errors[i])

    # close the writer
    writer.close()

    # env.scene.update_initial_state()
    # env.scene.reset()
    # breakpoint()
    og.sim.load_state(init_state)

# save episode errors
with open(f"{folder_path}/episode_errors.pkl", "wb") as f:
    pickle.dump(episode_errors, f)

print("done")


og.shutdown()
