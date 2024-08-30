import os
import yaml
import pickle
import cv2
import h5py
import time

import numpy as np
import matplotlib.pyplot as plt
import omnigibson as og
import omnigibson.lazy as lazy

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from filelock import FileLock
from PIL import Image
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController, draw_line
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory
from arpit_utils import save_video

def grasp_primitive(action_primitives, env, robot, episode_memory, arr):
    gripper_closed = False
    step = 0
    grasp_change_inds = []
    
    # Set grasp pose
    # global_pose3 = (np.array([-0.30722928, -0.79808354,  0.4792919 ]), np.array([-0.59315057,  0.37549363,  0.63922846,  0.3139489]))
    # right hand 1: (array([ 0.50431009, -0.25087801,  0.50123985]), array([ 0.57241185,  0.58268626, -0.41505368,  0.4006892 ]))
    org_pos, org_quat = np.array([ 0.50431009, -0.25087801,  0.46123985]), np.array([ 0.57241185,  0.58268626, -0.41505368,  0.4006892 ])

    # # get random pose in a cube
    # x_pos = np.random.uniform(0.3, 0.65)
    # y_pos = np.random.uniform(-0.45, 0.0)
    # z_pos = np.random.uniform(0.45, 0.75)
    # new_pos = np.array([x_pos, y_pos, z_pos])
    # # new_pos = np.array([0.4, -0.2, 0.7])


    # add noise to the position
    new_pos = org_pos + np.random.uniform(-0.02, 0.02, 3)
    # new_pos = org_pos + np.concatenate((np.random.uniform(-0.25, 0.25, 2), np.random.uniform(-0.05, 0.15, 1)))
    # new_pos = org_pos + np.concatenate((np.random.uniform(-0.15, 0.15, 2), np.random.uniform(0.0, 0.1, 1)))

    org_rotvec = np.array(R.from_quat(org_quat).as_rotvec())
    org_rotvec_angle = np.linalg.norm(org_rotvec)
    org_rotvec_axis = org_rotvec / org_rotvec_angle
    # add noise to the orientation
    new_axis = org_rotvec_axis + np.random.uniform(-0.1, 0.1, 3)
    new_axis /= np.linalg.norm(new_axis)
    new_angle = np.random.normal(org_rotvec_angle, 0.1)
    new_rotvec = new_axis * new_angle
    new_quat = R.from_rotvec(new_rotvec).as_quat()
    # print("------------", new_quat)

    # print("org_axis, org_angle: ", org_rotvec_axis, org_rotvec_angle)
    # print("new_axis, new_angle: ", new_axis, new_angle)

    # # For colleting data for random grasps
    # lis = np.arange(0, 400)
    # num_inds = np.random.randint(1,4)
    # grasp_change_inds = np.random.choice(lis, num_inds)
    # # print("grasp_change_inds: ", grasp_change_inds)

    # 1. Move to pregrasp pose
    pre_grasp_pose = (np.array(new_pos) + np.array([0.0, 0.0, 0.1]), np.array(new_quat))
    step, gripper_closed = execute_controller(action_primitives._move_hand_linearly_cartesian(pre_grasp_pose, in_world_frame=False, stop_if_stuck=False, ignore_failure=True, episode_memory=episode_memory, gripper_closed=gripper_closed),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    

    # 2. Move to grasp pose
    grasp_pose = (np.array(new_pos), np.array(new_quat))
    step, gripper_closed = execute_controller(action_primitives._move_hand_linearly_cartesian(grasp_pose, in_world_frame=False, stop_if_stuck=False, ignore_failure=True, episode_memory=episode_memory, gripper_closed=gripper_closed),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    
    # 3. Perform grasp
    gripper_closed = True
    action = action_primitives._empty_action()
    # left hand [11] ; right hand [18]
    action[18] = -1
    execute_controller([action], env, robot, gripper_closed, episode_memory, step, grasp_change_inds)
    # step the simulator a few steps to let the gripper close completely
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    # dump_to_memory(env, robot, episode_memory)    
    # episode_memory.add_action('actions', action[12:19])


    # 4. Move to a random pose in a neighbourhood
    # temp_pose = (org_pos + np.array([0.0, 0.0, 0.2]), org_quat)
    x = np.random.uniform(org_pos[0] - 0.2, org_pos[0] + 0.2)
    y = np.random.uniform(org_pos[1] - 0.2, org_pos[1] + 0.2)
    z = np.random.uniform(org_pos[2] + 0.2, org_pos[2] + 0.4)
    neighbourhood_pose = (np.array([x, y, z]), grasp_pose[1])
    # print("new_pos: ", new_pose[0])
    step, gripper_closed = execute_controller(action_primitives._move_hand_linearly_cartesian(neighbourhood_pose, in_world_frame=False, stop_if_stuck=False, ignore_failure=True, episode_memory=episode_memory, gripper_closed=gripper_closed),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    
    # Adding all 0 action for the last step
    episode_memory.add_action('actions', np.zeros(7, dtype=np.float64))


def custom_reset(env, robot, episode_memory):
    proprio = robot._get_proprioception_dict()
    curr_right_arm_joints = np.array(proprio['arm_right_qpos'])
    # default_joint_pos = robot.untucked_default_joint_pos[robot.arm_control_idx['right']]

    # print("proprio: ", proprio.keys())
    noise_1 = np.random.uniform(-0.2, 0.2, 3)
    noise_2 = np.random.uniform(-0.1, 0.1, 4)
    noise = np.concatenate((noise_1, noise_2))
    # print("arm_qpos.shape, noise.shape: ", curr_right_arm_joints.shape, noise.shape)
    right_hand_joints_pos = curr_right_arm_joints + noise 

    # Reset environment and robot
    env.reset()
    robot.reset(right_hand_joints_pos=right_hand_joints_pos)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def main():
    # Initializations
    np.random.seed(1451)

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "scale": [0.1, 0.05, 0.1],
            # "size": 0.05,
            "position": [-0.5, -0.7, 0.5],
            "orientation": [0.0004835024010390043,
                        -0.00029672126402147114,
                        -0.11094563454389572,
                        0.9938263297080994]
        },
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
    scene = env.scene
    robot = env.robots[0]

    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # save_folder = 'dynamics_model_dataset'
    save_folder = 'dynamics_model_dataset_seg_test'
    os.makedirs(save_folder, exist_ok=True)

    # Obtain the number of episodes
    episode_number = 0
    if os.path.isfile(f'{save_folder}/dataset.hdf5'):
        with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
            with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
                episode_number = len(file['data'].keys())
                print("episode_number: ", episode_number)

    for i in range(5):
        print(f"---------------- Episode {i} ------------------")
        start_time = time.time()
        episode_memory = Memory()

        arr = custom_reset(env, robot, episode_memory)

        for _ in range(50):
            og.sim.step()

        # # save the start simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_start.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
        #     pickle.dump(arr, f)
        
        grasp_primitive(action_primitives, env, robot, episode_memory, arr)
        # episode_memory.dump(f'{save_folder}/dataset.hdf5')

        # # save the end simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_end.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
        #     pickle.dump(arr, f)

        # # save video of the episode
        # save_video(np.array(episode_memory.data['observations']['rgb']), save_folder)

        del episode_memory
        episode_number += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Episode {episode_number}: execution time: {elapsed_time:.2f} seconds")
            
        

if __name__ == "__main__":
    main()
