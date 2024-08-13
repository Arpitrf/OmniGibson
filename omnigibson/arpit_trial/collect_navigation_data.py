import os
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from argparse import ArgumentParser
import time
from scipy.spatial.transform import Rotation as R
import h5py
from filelock import FileLock
import pprint

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController, draw_line
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

from memory import Memory
from arpit_utils import save_video

def obtain_gripper_obj_seg(img, img_info):
    # img = f[f'data/{k}/observations/seg_instance_id'][0]
    # img_info = np.array(f[f'data/{k}/observations_info']['seg_instance_id']).astype(str)[0]
    parts_of_concern = [  
        '/World/robot0/gripper_right_link/visuals',
        '/World/robot0/gripper_right_right_finger_link/visuals',
        '/World/robot0/gripper_right_left_finger_link/visuals',
        '/World/coffee_table_fqluyq_0/base_link/visuals',
        '/World/box/base_link/visuals'
    ]
    ids_of_concern = []
    for key, val in img_info.items():
        # print("val: ", val)
        if val in parts_of_concern:
            ids_of_concern.append(int(key))
    
    # print("ids_of_concern: ", ids_of_concern)
    new_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # print("img[i][j]: ", img[i][j], type(int(img[i][j])), type(ids_of_concern[0]))
            if int(img[i][j]) not in ids_of_concern:
                # print(int(img[i][j]))
                new_img[i][j] = 0
    return new_img

def dump_to_memory(env, robot, episode_memory):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()

    for k in obs['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs['robot0']['robot0:eyes:Camera:0'][k])
    # add gripper+object seg
    gripper_obj_seg = obtain_gripper_obj_seg(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'], obs_info['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
    episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info['robot0']['robot0:eyes:Camera:0'][k])
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k])

    is_grasping = robot.custom_is_grasping()
    is_contact = detect_robot_collision_in_sim(robot)

    episode_memory.add_extra('grasps', is_grasping)
    episode_memory.add_extra('contacts', is_contact)

def custom_reset(env, robot, episode_memory):
    proprio = robot._get_proprioception_dict()
    curr_right_arm_joints = np.array(proprio['arm_right_qpos'])
    # default_joint_pos = robot.untucked_default_joint_pos[robot.arm_control_idx['right']]

    # print("proprio: ", proprio.keys())
    noise_1 = np.random.uniform(-0.2, 0.2, 3)
    noise_2 = np.random.uniform(-0.1, 0.1, 4)
    noise = np.concatenate((noise_1, noise_2))
    # print("arm_qpos.shape, noise.shape: ", curr_right_arm_joints.shape, noise.shape)
    # right_hand_joint_pos = curr_right_arm_joints + noise 
    right_hand_joint_pos = curr_right_arm_joints  

    # Reset environment and robot
    env.reset()
    robot.reset(right_hand_joint_pos=right_hand_joint_pos)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    # add to memory
    obs, obs_info = env.get_obs()

    # remove later
    arr = []
    # seg_instance_id = obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id']
    # fig, ax = plt.subplots(2,2)
    # ax[0][0].imshow(obs['robot0']['robot0:eyes:Camera:0']['rgb'])
    # ax[0][1].imshow(obs['robot0']['robot0:eyes:Camera:0']['seg_semantic'])
    # ax[1][0].imshow(obs['robot0']['robot0:eyes:Camera:0']['seg_instance'])
    # ax[1][1].imshow(seg_instance_id)
    # plt.show()  
    # n = 5
    # arr = []
    # # iterating till the range
    # for i in range(0, n):
    #     ele = int(input())
    #     # adding the element
    #     arr.append(ele)
    # for i in range(seg_instance_id.shape[0]):
    #     for j in range(seg_instance_id.shape[1]):
    #         if seg_instance_id[i][j] not in arr:
    #             seg_instance_id[i][j] = 0
    
    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()

    for k in obs['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs['robot0']['robot0:eyes:Camera:0'][k])
    # add gripper+object seg
    gripper_obj_seg = obtain_gripper_obj_seg(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'], obs_info['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
    episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)

    for k in obs_info['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info['robot0']['robot0:eyes:Camera:0'][k])
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k])

    is_grasping = robot.custom_is_grasping()
    is_contact = detect_robot_collision_in_sim(robot)
    episode_memory.add_extra('grasps', is_grasping)
    episode_memory.add_extra('contacts', is_contact)

    return arr



def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma")
    # parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--hand', type=str, default='left')
    parser.add_argument('--save_traj', action='store_true', default=False)
    return parser
    

def execute_controller(ctrl_gen, env, robot, gripper_closed, episode_memory, step, grasp_change_inds, arr=None):
    actions = []
    counter = 0
    # print("type(ctrl_gen): ", ctrl_gen)
    for action in ctrl_gen:
        print("action: ", action)
        if action == 'Done':
            # save everything to data
            obs, obs_info = env.get_obs()

            # seg_instance_id = obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id']
            # print("arr: ", arr)
            # if arr is not None:
            #     for i in range(seg_instance_id.shape[0]):
            #         for j in range(seg_instance_id.shape[1]):
            #             if seg_instance_id[i][j] not in arr:
            #                 obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'][i][j] = 0
            # fig, ax = plt.subplots(2,2)
            # ax[0][0].imshow(obs['robot0']['robot0:eyes:Camera:0']['rgb'])
            # ax[0][1].imshow(obs['robot0']['robot0:eyes:Camera:0']['seg_semantic'])
            # ax[1][0].imshow(obs['robot0']['robot0:eyes:Camera:0']['seg_instance'])
            # ax[1][1].imshow(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
            # plt.show()

            proprio = robot._get_proprioception_dict()
            # add eef pose and base pose to proprio
            proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
            proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
            proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()

            for k in obs['robot0']['robot0:eyes:Camera:0'].keys():
                episode_memory.add_observation(k, obs['robot0']['robot0:eyes:Camera:0'][k])
            # add gripper+object seg
            gripper_obj_seg = obtain_gripper_obj_seg(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'], obs_info['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
            episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)

            for k in obs_info['robot0']['robot0:eyes:Camera:0'].keys():
                episode_memory.add_observation_info(k, obs_info['robot0']['robot0:eyes:Camera:0'][k])
            for k in proprio.keys():
                episode_memory.add_proprioception(k, proprio[k])

            is_grasping = robot.custom_is_grasping()
            is_contact = detect_robot_collision_in_sim(robot)

            episode_memory.add_extra('grasps', is_grasping)
            episode_memory.add_extra('contacts', is_contact)
            
            continue

        # print("action: ", action[12:18], np.linalg.norm(action[12:18]))
        # complete_action = action
        # [12,13,14]: delta_pos; [15,16,17]: delta_axis_angle; [18]: grasp(-1 is close and 1 is open) 
        # preprocessed_arm_action = action[12:19]
        # command = robot._controllers['arm_right']._preprocess_command(action[12:18])
        # processed_arm_action = np.concatenate((command, action[18:19]))
        # print("processed_arm_action: ", processed_arm_action.shape)
        # print("preprocessed_arm_action: ", preprocessed_arm_action.shape)
        # print("complete_action: ", complete_action.shape)
        
        # actions.append(action[5:11].tolist())

        wait = False
        # To collect data for random grasps
        # print("step, graspstep, )
        if step in grasp_change_inds:
            gripper_closed = not gripper_closed
            wait = True

        if gripper_closed:
            # action[11] = -1
            action[18] = -1
        else: 
            # action[11] = 1
            action[18] = 1
        
        env.step(action)
        if wait:
            for _ in range(60):
                og.sim.step()
    
        counter += 1
        step += 1
    
    print("total steps: ", counter)
    return step, gripper_closed

def navigate_primitive(action_primitives, env, robot, episode_memory, arr):
    obj = env.scene.object_registry("name", "apple")
    gripper_closed = False
    step = 0
    grasp_change_inds = []

    # execute_controller(action_primitives._navigate_to_obj(obj),
    #                      env,
    #                      robot,
    #                      gripper_closed,
    #                      episode_memory,
    #                      step,
    #                      grasp_change_inds)
    
    target_pos, target_orn_quat = np.array([-4.9999990e-02, -3.5000005e-01, -1.8534674e-07]), np.array([-5.8735580e-08,  2.1243411e-08, -7.6604444e-01,  6.4278764e-01])
    target_orn_euler = R.from_quat(target_orn_quat).as_euler('XYZ')
    pose2d = np.array([target_pos[0], target_pos[1], target_orn_euler[2]])
    # add some noise:
    noise_orn = np.random.uniform(-0.35, 0.35)
    noise_pos = np.random.uniform(-0.1, 0.1, 2)
    # # target_orn_euler[2] -= 0.35
    # target_pos[0] -= 0.1
    pose2d += np.array([noise_pos[0], noise_pos[1], noise_orn])    
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(pose2d),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    
def main():
    # Initializations
    np.random.seed(1451)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    # config["objects"] = [
    #     # {
    #     #     "type": "PrimitiveObject",
    #     #     "name": "box",
    #     #     "primitive_type": "Cube",
    #     #     "rgba": [1.0, 0, 0, 1.0],
    #     #     "scale": [0.1, 0.05, 0.1],
    #     #     # "size": 0.05,
    #     #     "position": [-0.5, -0.7, 0.5],
    #     # },
    #     {
    #         "type": "DatasetObject",
    #         "name": "table",
    #         "category": "breakfast_table",
    #         "model": "rjgmmy",
    #         "scale": [0.3, 0.3, 0.3],
    #         "position": [-0.7, 0.5, 0.2],
    #         "orientation": [0, 0, 0, 1]
    #     },
    #     {
    #         "type": "DatasetObject",
    #         "name": "apple",
    #         "category": "apple",
    #         "model": "agveuv",
    #         "position": [-0.5, -0.7, 0.5],
    #         "orientation": [0, 0, 0, 1]
    #     },
    # ]

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "conference_table",
            "model": "qzmjrj", 
            "position": [(-0.5 + 1) - 0.5, (-0.7 - 1.5) + 0.5, 0.4],
            "scale": [1, 1, 0.6],
            "orientation": R.from_euler("xyz", [0, 0, -3.14/2]).as_quat()
        },
        {
            "type": "DatasetObject",
            "name": "apple", 
            "category": "apple",
            "model": "omzprq", 
            "position": [-0.1, -0.9, 0.6]
        },
        {
            "type": "DatasetObject",
            "name": "plate",
            "category": "plate",
            "model": "pkkgzc",
            "position": [(-0.75 + 1) - 0.5, (-0.7 - 1.5) + 0.5, 0.6]
        }
    ]

    pprint.pprint(config)

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # # Create teleop controller
    # action_generator = KeyboardRobotController(robot=robot)
    # # Register custom binding to reset the environment
    # action_generator.register_custom_keymapping(
    #     key=lazy.carb.input.KeyboardInput.R,
    #     description="Reset the robot",
    #     callback_fn=lambda: env.reset(),
    # )
    # # Print out relevant keyboard info if using keyboard teleop
    # action_generator.print_keyboard_teleop_info()

    # save_folder = 'dynamics_model_dataset'
    save_folder = 'navigation_dataset_seg'
    os.makedirs(save_folder, exist_ok=True)

    # og.sim.restore('dynamics_model_data/episode_00000_start.json')
    # # step the simulator a few times 
    # for _ in range(1000):
    #     og.sim.step()

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

        base_pos, base_orn = robot.get_position_orientation()
        print("base_pos, base_orn: ", base_pos, base_orn)
        euler = R.from_quat(base_orn).as_euler('XYZ', degrees=True)
        rotvec = R.from_quat(base_orn).as_rotvec()
        print("euler: ", euler)
        print("rotvec: ", rotvec / np.linalg.norm(rotvec), np.linalg.norm(rotvec))

        # # save the start simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_start.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
        #     pickle.dump(arr, f)
        
        # grasp_primitive(action_primitives, env, robot, episode_memory, arr)
        navigate_primitive(action_primitives, env, robot, episode_memory, arr)
        # episode_memory.dump(f'{save_folder}/dataset.hdf5')

        # # save the end simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_end.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
        #     pickle.dump(arr, f)

        # # save video of the episode
        # save_video(np.array(episode_memory.data['observations']['rgb']), save_folder)

        for _ in range(150):
            og.sim.step()

        del episode_memory
        episode_number += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Episode {episode_number}: execution time: {elapsed_time:.2f} seconds")
            
        

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
