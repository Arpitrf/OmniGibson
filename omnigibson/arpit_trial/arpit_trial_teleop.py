import os
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma")
    # parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--hand', type=str, default='left')
    parser.add_argument('--save_traj', action='store_true', default=False)
    return parser

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def print_dict_keys(dictionary, indent=0):
    for key, value in dictionary.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_dict_keys(value, indent + 4)

def custom_reset(env, robot):
    proprio = robot._get_proprioception_dict()
    curr_right_arm_joints = np.array(proprio['arm_right_qpos'])
    # default_joint_pos = robot.untucked_default_joint_pos[robot.arm_control_idx['right']]

    # print("proprio: ", proprio.keys())
    noise_1 = np.random.uniform(-0.2, 0.2, 3)
    # noise_2 = np.random.uniform(-0.1, 0.1, 4)
    noise_2 = np.random.uniform(-0.01, 0.01, 4)
    noise = np.concatenate((noise_1, noise_2))
    # print("arm_qpos.shape, noise.shape: ", curr_right_arm_joints.shape, noise.shape)
    # right_hand_joints_pos = default_joint_pos + noise 
    # right_hand_joints_pos = default_joint_pos
    right_hand_joints_pos = curr_right_arm_joints + noise

    scene_initial_state = env.scene._initial_state
    # for manipulation
    base_pos = np.array([-0.05, -0.4, 0.0])
    base_x_noise = np.random.uniform(-0.15, 0.15)
    base_y_noise = np.random.uniform(-0.15, 0.15)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = base_pos
    
    base_yaw = -120
    base_yaw_noise = np.random.uniform(-15, 15)
    base_yaw += base_yaw_noise
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    # print("r_quat: ", r_quat)

    # Randomizing head pose
    # default_head_joints = np.array([-0.20317451, -0.7972661])
    default_head_joints = np.array([-0.5031718015670776, -0.9972541332244873])
    noise_1 = np.random.uniform(-0.1, 0.1, 1)
    noise_2 = np.random.uniform(-0.1, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints = default_head_joints + noise
    # print("Head joint positions: ", head_joints)

    # Reset environment and robot
    env.reset()
    robot.reset(right_hand_joints_pos=right_hand_joints_pos, head_joints_pos=head_joints)
    # robot.reset()

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    
    It loads Rs_int with a Fetch robot, and the robot picks and places a bottle of cologne.
    """
    # Initializations
    np.random.seed(1)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "size": 0.05,
            "position": [-0.3, -0.8, 0.5],
        },
        # {
        #     "type": "DatasetObject",
        #     "name": "cologne",
        #     "category": "bottle_of_cologne",
        #     "model": "lyipur",
        #     "position": [-0.3, -0.8, 0.5],
        #     "orientation": [0, 0, 0, 1]
        # },
        # {'type': 'DatasetObject', 'name': 'obj', 'category': 'zucchini', 'model': 'dybtwb', 'position': [-0.3, -0.8, 0.5]},
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1]
        }
    ]

    # Load the environment
    env = og.Environment(configs=config)
    # og.camera_mover.clear
    scene = env.scene
    robot = env.robots[0]

    # # Allow user to move camera more easily
    # og.sim.enable_viewer_camera_teleoperation()

    # Teleop robot
    custom_reset(env, robot)
    
    # add orientations and base mvmt
    action_keys = ['P', 'SEMICOLON', 'RIGHT_BRACKET', 'LEFT_BRACKET', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'Y', 'B', 'N', 'O', 'U', 'C', 'V']
    teleop_traj = []
    
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    max_steps = -1
    step = 0
    while step != max_steps:
        action, keypress_str = action_generator.get_teleop_action()
        # print("keypress_str: ", step, keypress_str)
        # print("action: ", action, len(action))
        if keypress_str == 'TAB':
            break
        obs, _, _, _ = env.step(action=action)
        step += 1
        # if keypress_str in action_keys:
        #     print("action: ", action)
        #     teleop_traj.append(action)
        if step % 50 == 0:
            is_collision = detect_robot_collision_in_sim(robot, ignore_obj_in_hand=True)
            print("is_collision: ", is_collision)            
            # print("proprio_dict: ", robot._get_proprioception_dict().keys())
            # print("joint_qpos: ", robot._get_proprioception_dict()['joint_qpos'], robot._get_proprioception_dict()['joint_qpos'].shape)
            # print("camera_qpos: ", robot._get_proprioception_dict()['camera_qpos'], robot._get_proprioception_dict()['camera_qpos'].shape)
    

    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()