import os
import yaml
import  pdb
import pickle
import cv2
import imageio
import random

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

from motion_utils import MotionUtils
from memory import Memory
from collision_failure_model import CollisionFailureModel
from utils import correct_gripper_friction, check_success


num_samples = 10
num_top_samples = 3
epochs = 10
success = False

mu_x = np.zeros(3) + 0.03  # Example: 2-dimensional problem
sigma_x = np.eye(3) * 0.003
mu_y = np.zeros(3) # Example: 2-dimensional problem
sigma_y = np.eye(3) * 0.001
mu_z = np.zeros(3)  # Example: 2-dimensional problem
sigma_z = np.eye(3) * 0.001

temp_prior = th.tensor([
    [ 0.   ,  0.,    -0.301,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.104,  0.334,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.267,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.1,   -0.,    -0.,    -0.,    -1.   ],
    [ 0.   ,  0.,    -1.456,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.459,  0.034,  0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,    -0.109,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.035, -0.016,  0.105, -0.015, -0.103, -0.065, -1.   ],
    # [ 0.   ,  0.,     0.,     0.044, -0.017,  0.142, -0.016, -0.095, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.074, -0.017,  0.092, -0.016, -0.095, -0.067, -1.   ],
    # [ 0.   ,  0.,     0.,     0.038, -0.017,  0.144, -0.016, -0.103, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.078, -0.017,  0.054, -0.016, -0.103, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
])

def expl(t, actions, motion_utils, robot, env, traj_length, shelf_pos_orn, start_idx, collision_failure_model=None, grasp_mode=None):
    
    if t == traj_length:
        print("Reached end of recursion")
        # open gripper and see
        a = th.zeros(10)
        a[-1] = 1.0
        # input("open gripper action")
        retval = motion_utils.safe(a, use_hack=True, collision_failure_model=collision_failure_model, grasp_mode=grasp_mode)
        return retval
    
    for action in actions:
        print("--- time step, action: ", t, action[t][3:6])
        ee_pose_before = robot.get_relative_eef_pose(arm='right')
        joint_pos_before = robot.get_joint_positions()[robot.arm_control_idx["right"]]
        sim_state_before = og.sim.dump_state()
        if motion_utils.safe(action[t], collision_failure_model=collision_failure_model, grasp_mode=grasp_mode):
            # In the current implementation I am performing the action (move_primitive) inside the safe action. This will change later.
            all_failed = expl(t+1, actions, motion_utils, robot, env, traj_length, shelf_pos_orn, start_idx, collision_failure_model, grasp_mode)

            # if task success
            if check_success(env, robot):
                print("Task succeeded!")
                all_failed = False 
                return all_failed
            
            # undo the last action. For now try making it go back to exact joint positions
            motion_utils.undo_action(t, action)        
                    
            ee_pose_after = robot.get_relative_eef_pose(arm='right')
            pos_error = np.linalg.norm(ee_pose_after[0] - ee_pose_before[0])
            orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], ee_pose_before[1])
            # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
            # print("joint_pos_before: ", joint_pos_before)
            joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            # print("joint_pos after rewind: ", joint_pos_after)
            need_reset = any(abs(joint_pos_before - joint_pos_after) > 0.1)
            print("need to call sim.load_state?: ", need_reset)
            if need_reset:
                og.sim.load_state(sim_state_before)
                for _ in range(30):
                    og.sim.step()
                joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                # print("joint_pos after reset: ", joint_pos_after)
            # input("Undid the action. Press enter to continue")

            # reset the shelf in case it has moved
            shelf = env.scene.object_registry("name", "shelf")
            shelf.set_position_orientation(shelf_pos_orn[0], shelf_pos_orn[1])
            for _ in range(10):
                og.sim.step()

            if all_failed:
                # sample t+1 actions again
                actions = sample_actions(t+1, actions, traj_length, start_idx)

    all_failed = True
    return all_failed 

def sample_actions(t, actions, traj_length, start_idx):
    if t == traj_length:
        print("No action sampling needed.")
        return actions
    
    x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)
    
    actions_org = temp_prior.clone()
    actions_org = actions_org[None, start_idx:-1]
    actions_org = actions_org.repeat(num_samples, 1, 1)
    actions[:, t:, 3:6] = actions_org[:, t:, 3:6] + episode_pos_noise[:, t:]
    return actions


def get_episode_reward(robot):
    target_pos = [1.1888, -0.1884,  0.8387]
    curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
    dist = np.linalg.norm(target_pos - curr_pos)
    return -dist

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


def main():
    set_all_seeds(seed=13)
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"] = dict()
    config["scene"]["type"] = "Scene"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = f"outputs/run_{current_time}"
    os.makedirs(folder_path, exist_ok=True)

    # Create and load this object into the simulator
    rot_euler = [0.0, 0.0, 180.0]
    rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
    box_euler = [0.0, 0.0, 0.0]
    box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "shelf",
            "category": "shelf",
            "model": "eniafz",
            "position": [1.5, 0, 1.0],
            "scale": [2.0, 2.0, 1.0],
            "orientation": rot_quat,
        },   
        {
            "type": "DatasetObject",
            "name": "coffee_table",
            "category": "coffee_table",
            "model": "fqluyq",
            # "scale": [0.3, 0.3, 0.3],
            "position": [0, 0.6, 0.3],
            "orientation": [0, 0, 0, 1]
        },
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "scale": [0.1, 0.05, 0.1],
            # "size": 0.05,
            # "mass": 1e-6,
            "position": [0.1, 0.5, 0.5],
            "orientation": box_quat
        },
    ]

    env = og.Environment(configs=config)
    og.sim.restore(["moma_pick_and_place/episode_00000_start.json"])
    # og.sim.restore(["place_start.json"])

    scene = env.scene
    robot = env.robots[0]
    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    correct_gripper_friction(robot)

    # Modify object properties
    shelf = env.scene.object_registry("name", "shelf")
    # shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))
    # coffee_table = env.scene.object_registry("name", "coffee_table")
    # shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))
    # coffee_table.set_position_orientation(position=th.tensor([10.0, 10.0, 0.0]))
    shelf.root_link.mass = 1e3
    box = env.scene.object_registry("name", "box")
    box.root_link.mass = 1e-2
    shelf_pos_orn = shelf.get_position_orientation()

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        th.tensor([-0.7563,  1.1324,  1.0464]),
        th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
    )

    # for saving videos
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
    base_folder = f"{current_date}"
    time_folder = os.path.join(base_folder, current_time)
    folder_path = f"outputs_expl/{time_folder}"
    os.makedirs(folder_path, exist_ok=True)

    imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    output_path = f'{folder_path}/video.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)

    motion_utils = MotionUtils(env, robot, action_primitives, writer)
    motion_utils.custom_reset(env, robot)

    collision_failure_model = CollisionFailureModel()

    for _ in range(50):
        og.sim.step()


    # robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
    # for _ in range(30):
    #     og.sim.step()

    # # replay nav sub-tasks
    # traj_length = 8
    # start_idx = 0
    # for t in range(start_idx, traj_length):
    #     move_primitive(temp_prior[t])
    #     if t == 2:
    #         perform_grasp()


    # Try two modes
    modes = ["vertical", "horizontal"]

    episode_memory = Memory()
    primitive_steps_to_perform = np.arange(1, 6)

    for mode in modes:
        grasp_mode = mode
        # use primitives for the initial actions
        grasp_sim_state = motion_utils.first_primitive(primitive_steps_to_perform, episode_memory=episode_memory, grasp_mode=grasp_mode) 
        # breakpoint()

        traj_length = 3 # for the place subtask
        start_idx = 8

        x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
        y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
        z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
        episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                        np.expand_dims(y_noise, axis=1), 
                        np.expand_dims(z_noise, axis=1)), axis=1)

        actions = temp_prior.clone()
        actions = actions[None, start_idx:-1]
        actions = actions.repeat(num_samples, 1, 1)
        actions[:, :, 3:6] = actions[:, :, 3:6] + episode_pos_noise
        print("Start actions shape: ", actions.shape)

        all_failed = expl(t=0, actions=actions, motion_utils=motion_utils, robot=robot, env=env, traj_length=traj_length, shelf_pos_orn=shelf_pos_orn, start_idx=start_idx, collision_failure_model=collision_failure_model, grasp_mode=grasp_mode)

        if not all_failed:
            break

        if all_failed:
            og.sim.load_state(grasp_sim_state)

        primitive_steps_to_perform = np.array([2, 3, 4, 5])

    breakpoint()
    og.shutdown()


if __name__ == "__main__":
    main()