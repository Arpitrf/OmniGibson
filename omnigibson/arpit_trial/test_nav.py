import os
import yaml
import numpy as np
np.set_printoptions(suppress=True, precision=3)
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
import math

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController, draw_line
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.objects import DatasetObject

from memory import Memory
from arpit_utils import save_video

import cv2
import os

def create_video_from_images(image_folder, output_video, fps=30):
    # Get list of all files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # For OmniGibson videos
    # images.sort(key=lambda x: int(x.split("_")[1])) # For HaMer videos
    # images.pop(0)

    if not images:
        print("No .jpg images found in the folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'test_nav_outputs/{output_video}', 0, fps, (width, height))

    list = os.listdir(image_folder)
    # print(images)

    # Loop through all images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video {output_video} created successfully.")

# Example usage:
# image_folder = 'prior_test1_images'
# output_video = f'{image_folder}.avi'
# create_video_from_images(image_folder, output_video, fps=15)


def custom_reset(env, robot, episode_memory=None):
    proprio = robot._get_proprioception_dict()
    print("proprio.keys: ", proprio.keys())
    curr_right_arm_joints = np.array(proprio['arm_right_qpos'])
    # default_joint_pos = robot.untucked_default_joint_pos[robot.arm_control_idx['right']]

    # Randomizing right arm pose
    noise_1 = np.random.uniform(-0.3, 0.3, 3)
    noise_2 = np.random.uniform(-0.1, 0.1, 4)
    noise = np.concatenate((noise_1, noise_2))
    # print("arm_qpos.shape, noise.shape: ", curr_right_arm_joints.shape, noise.shape)
    right_hand_joints_pos = curr_right_arm_joints + noise 
    # right_hand_joints_pos = curr_right_arm_joints  
    # print("right_hand_joints_pos: ", right_hand_joints_pos)

    # Randomizing head pose
    default_head_joints = np.array([-0.20317451, -0.7972661])
    noise_1 = np.random.uniform(-0.1, 0.1, 1)
    noise_2 = np.random.uniform(-0.1, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints = default_head_joints + noise
    # print("Head joint positions: ", head_joints)

    # Reset environment and robot
    # print("env initial state: ", env.scene._initial_state)
    scene_initial_state = env.scene._initial_state

    # for manipulation
    # scene_initial_state['object_registry']['robot0']['joints']['head_2_joint']['target_pos'] = np.array([-0.83])
    # scene_initial_state['object_registry']['robot0']['root_link']['pos'] = [-0.05, -0.4, 0.0]
    # r_euler = R.from_euler('z', -100, degrees=True) # or -120

    # Randomizing object pose

    # Randomizing object sizes
    table_z_scale = np.random.uniform(0.5, 0.8)
    table_x_scale = np.random.uniform(0.9, 1.1)
    table_y_scale = np.random.uniform(0.95, 1.05)
    # apple_z_pos = table_z_scale * 0.8       # table_z_scale=1 is 0.8 height for apple
    og.sim.stop()
    for o in env.scene.objects:
        if o.name == 'table':
            o.scale = np.array([table_x_scale, table_y_scale, table_z_scale])
        if o.name == 'apple':
            apple_scale = np.random.uniform(0.9, 1, 3)
            o.scale = apple_scale
            apple_pos = o.get_position()
            # target_pos = np.array([apple_pos[0], apple_pos[1], apple_z_pos])
            # print("target_pos: ", target_pos)
            # o.set_position_orientation(position=target_pos)
    og.sim.play()
    print("apple_pos: ", apple_pos)
    # for o in env.scene.objects:
    #     if o.name == 'apple':
    #         print("o.pos: ", o.get_position())
    #         input()

    # Randomizing starting pos of robot
    start_pos = np.array([-0.1, 2.2, 0.0])
    # start_pos = np.array([-0.05, 0.35, 0.0])
    # x_noise = np.random.uniform(-0.3, 0.8)
    # # x_noise = 0.7
    # start_pos[0] += x_noise
    # y_noise = np.random.uniform(-0.3, 0.5)
    # # y_noise = 0.5
    # start_pos[1] += y_noise
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = start_pos

    # Ramdomizing starting orientation of robot
    # yaw = np.random.uniform(-80, -120)
    yaw = 0
    r_euler = R.from_euler('z', yaw, degrees=True)
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

    env.reset()
    # TODO: Change this through the same interface (using scene_initial_state)
    robot.reset(right_hand_joints_pos=right_hand_joints_pos, head_joints_pos=head_joints)


    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()


def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma")
    # parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--hand', type=str, default='left')
    parser.add_argument('--save_traj', action='store_true', default=False)
    return parser
    
def main():
    # Initializations
    np.random.seed(1)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    # config["scene"]["scene_model"] = "Rs_int"
    # config["scene"]["load_object_categories"] = ["floors"]
    config["scene"] = {
        "type" : "Scene",
        "floor_plan_visible" : True
    }
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
            "position": [-0.2, -0.9, 0.6]
        },
        {
            "type": "DatasetObject",
            "name": "plate",
            "category": "plate",
            "model": "pkkgzc",
            "position": [(-0.75 + 1) - 0.5, (-0.7 - 1.5) + 0.5, 0.5]
        }
    ]

    pprint.pprint(config)

    save_images = True
    video_name = "nav_test9"
    save_image_dir = f"test_nav_outputs/{video_name}_images"
    fps = 5
    frame_step = 30 // fps
    counter = 0
    if save_images:
        os.makedirs(save_image_dir, exist_ok=True)
    arnav_method = False
    arpit_method = False
    third_method = True

    data = np.load(f"prior_npz_files/{video_name}_prior_results.npz")
    body_trans = data["body_positions"]
    body_orient = data["body_orientations"]
    hand_positions = data["hand_positions"]
    hand_rotations = data["hand_orientations"]

    print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
    print(f"hand_rotations shape: {hand_rotations.shape}")
    print(f"body_trans shape: {body_trans.shape}")
    print(f"body_orient shape: {body_orient.shape}\n\n\n")

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    controller.arm = "right"

    custom_reset(env, robot)

    # prev_body_pos, prev_body_orn = robot.get_position_orientation()
    # print(f"\n\nStarting Body Position: {prev_body_pos, prev_body_orn}\n\n")
 
    # prev_hand_pos, prev_hand_orn = robot.eef_links["right"].get_position_orientation()
    # prev_hand_pos, prev_hand_orn = controller._get_pose_in_robot_frame([prev_hand_pos, prev_hand_orn])
    # print(f"\n\nStarting Hand Position: {prev_hand_pos, prev_hand_orn}\n\n")

    # Obtain init robot pose from OG
    init_robot_pos_in_world, init_robot_orn_in_world = robot.get_position_orientation()
    init_robot_orn_matrix_in_world = R.from_quat(init_robot_orn_in_world).as_matrix()
    init_robot_pose = np.eye(4)
    init_robot_pose[:3, :3] = init_robot_orn_matrix_in_world
    init_robot_pose[:3, 3] = np.transpose(init_robot_pos_in_world)

    # obtain init human pose from slahmr-hamer
    init_human_pos, init_human_orn = body_trans[0], body_orient[0]
    init_human_orn_matrix = R.from_euler("xyz", init_human_orn).as_matrix()
    init_human_pose = np.eye(4)
    init_human_pose[:3, :3] = init_human_orn_matrix
    init_human_pose[:3, 3] = np.transpose(init_human_pos)

    prev_human_pose = init_human_pose.copy()
    
    # If we do w.r.t init pose and not prev pose
    prev_delta_pos_wrt_robot = np.array([0.0, 0.0, 0.0])
    
    prev_delta_orientation_wrt_init_human_pose = 0.0

    for i in range (frame_step, hand_positions.shape[0], frame_step):
    # for i in range (0, hand_positions.shape[0]):
        print("Step: ", i)
        current_human_pos, current_human_orn = body_trans[i], body_orient[i]
        # prev_human_pos, prev_human_orn = body_trans[i - frame_step], body_orient[i - frame_step]

        if third_method:
            # Getting current human pose from slahmr-hamer
            current_human_orn_matrix = R.from_euler("xyz", current_human_orn).as_matrix()
            current_human_pose = np.eye(4)
            current_human_pose[:3, :3] = current_human_orn_matrix
            current_human_pose[:3, 3] = np.transpose(current_human_pos)
            # If we do w.r.t init pose and not prev pose
            current_human_pose_wrt_init_human_pose = np.dot(np.linalg.inv(init_human_pose), current_human_pose)
            print("bi wrt b1: ", current_human_pose_wrt_init_human_pose[:3, 3])

            # # Getting delta human positions (w.r.t previous human frame): This will be the delta positions for the robot base as well 
            # # (as we assume that starting pose of robot and human is similar)
            # current_human_pose_wrt_prev_human_pose = np.dot(np.linalg.inv(prev_human_pose), current_human_pose)
            # x_pos_delta = current_human_pose_wrt_prev_human_pose[2, 3]
            # y_pos_delta = current_human_pose_wrt_prev_human_pose[0, 3]
            # If we do w.r.t init pose and not prev pose
            current_human_pose_wrt_init_human_pose = np.dot(np.linalg.inv(init_human_pose), current_human_pose)
            x_pos_delta = current_human_pose_wrt_init_human_pose[2, 3]
            y_pos_delta = current_human_pose_wrt_init_human_pose[0, 3]
            
            delta_pos_wrt_robot = np.array([x_pos_delta, y_pos_delta, 0])
            # If we do w.r.t init pose and not prev pose
            current_delta_pos_wrt_robot= delta_pos_wrt_robot.copy()
            delta_pos_wrt_robot = delta_pos_wrt_robot - prev_delta_pos_wrt_robot
            print("curr, prev delta_pos_wrt_robot: ", delta_pos_wrt_robot, prev_delta_pos_wrt_robot)
            print("delta_pos_wrt_robot: ", delta_pos_wrt_robot)
            
            # Now we need to transform the delta position in robot frame to world frame. Obtain current robot pose w.r.t world
            current_robot_pos_wrt_world, current_robot_orn_wrt_world = robot.get_position_orientation()
            current_robot_orn_matrix_wrt_world = R.from_quat(current_robot_orn_wrt_world).as_matrix()
            current_robot_pose_wrt_world = np.eye(4)
            current_robot_pose_wrt_world[:3, :3] = current_robot_orn_matrix_wrt_world
            current_robot_pose_wrt_world[:3, 3] = np.transpose(current_robot_pos_wrt_world)
            print("current_robot_pos_wrt_world: ", current_robot_pos_wrt_world)

            # Obtain delta positions w.r.t world
            delta_pos_wrt_world = np.dot(current_robot_orn_matrix_wrt_world, delta_pos_wrt_robot)
            target_pos_wrt_world = current_robot_pos_wrt_world + delta_pos_wrt_world[:3]
            print("target_pos_wrt_world: ", target_pos_wrt_world)
            
            # getting orientations
            rotmat = current_human_pose_wrt_init_human_pose[:3, :3]
            unit_vector = np.array([1., 0., 0.])
            direction_vector = np.matmul(rotmat, unit_vector)
            delta_orientation_wrt_init_human_pose = math.atan2(direction_vector[2], direction_vector[0])
            print(f"delta_orientation_wrt_init_human_pose: {np.rad2deg(delta_orientation_wrt_init_human_pose)} degrees")
            current_delta_orientation_wrt_init_human_pose = delta_orientation_wrt_init_human_pose
            delta_orientation = current_delta_orientation_wrt_init_human_pose - prev_delta_orientation_wrt_init_human_pose
            current_robot_orientation = np.array(R.from_quat(robot.get_orientation()).as_euler('xyz'))[2]
            final_robot_orientation_wrt_world = current_robot_orientation - delta_orientation
            print("current, delta robot orn: ", np.rad2deg(current_robot_orientation), np.rad2deg(delta_orientation))
            # input()
            
            # Final input to the base controller
            target_pose = np.array([target_pos_wrt_world[0], target_pos_wrt_world[1], final_robot_orientation_wrt_world])
            # input()

            # Setting previous to current
            prev_delta_pos_wrt_robot = current_delta_pos_wrt_robot
            prev_human_pose = current_human_pose
            prev_delta_orientation_wrt_init_human_pose = current_delta_orientation_wrt_init_human_pose

        
        if arpit_method: 
            # Getting current body pose w.r.t first body pose
            current_human_orn_matrix = R.from_euler("xyz", current_human_orn).as_matrix()
            current_human_pose = np.eye(4)
            current_human_pose[:3, :3] = current_human_orn_matrix
            current_human_pose[:3, 3] = np.transpose(current_human_pos)
            current_human_pose_wrt_init_human_pose = np.dot(np.linalg.inv(init_human_pose), current_human_pose)
            print("bi wrt b1: ", current_human_pose_wrt_init_human_pose[:3, 3])
            # print("curr_pos - init_pos: ", current_human_pos - human_pos_1)

            # Getting the target pose
            # maybe need a fixed transformation between human hip joint and robot base 
            fixed_transform = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
            target_pose = np.dot(init_robot_pose, fixed_transform)
            target_pose = np.dot(target_pose, current_human_pose_wrt_init_human_pose)
            target_pos = target_pose[:3, 3]
            target_orn_matrix = target_pose[:3, :3]
            target_orn_euler = R.from_matrix(target_orn_matrix).as_euler('xyz', degrees=True)
            target_yaw = target_orn_euler[2]
            curr_robot_pos, curr_robot_orn = robot.get_position_orientation()
            curr_robot_yaw = R.from_quat(curr_robot_orn).as_euler('xyz', degrees=True)[2]
            print("curr_pos, curr_yaw: ", curr_robot_pos, curr_robot_yaw)
            print("target_pos, target_yaw: ", target_pos, target_yaw)
            target_pose = [target_pos[0], target_pos[1], target_yaw]

        # Getting body deltas
        if arnav_method:
            delta_body_pos = np.subtract(current_human_pos, prev_pos)
            # delta_body_pos = np.array([delta_body_pos[0], delta_body_pos[2], delta_body_pos[1]]) # Switching the y and z values
            delta_body_pos = np.array([delta_body_pos[2], delta_body_pos[0], delta_body_pos[1]]) # Switching the y and z values
            target_body_pos = np.add(prev_body_pos, delta_body_pos)
            print("target, prev, delta pos: ", target_body_pos, prev_body_pos, delta_body_pos)

            rotmat = R.from_euler("xyz", current_human_orn).as_matrix()
            unit_vector = np.array([1., 0., 0.])
            direction_vector = np.matmul(rotmat, unit_vector)
            current_orientation = math.atan2(direction_vector[2], direction_vector[0])

            rotmat_prev = R.from_euler("xyz", prev_orn).as_matrix()
            unit_vector_prev = np.array([1., 0., 0.])
            direction_vector_prev = np.matmul(rotmat_prev, unit_vector_prev)
            # print("direction_vector_prev: ", direction_vector_prev)
            prev_orientation = math.atan2(direction_vector_prev[2], direction_vector_prev[0])

            delta_orientation = current_orientation - prev_orientation
            # print("curr, prev, delta ori: ", current_orientation, prev_orientation, delta_orientation)

            target_body_orn = R.from_quat(prev_body_orn).as_euler("xyz")
            target_body_orn[2] += delta_orientation

            # remove later
            # temp_body_orn = np.array(R.from_quat(robot.get_orientation()).as_euler('xyz', degrees=True))[2]
            # print("temp_body_orn: ", temp_body_orn)
            # input()
            # target_body_orn[2] = np.deg2rad(-90)
            target_pose = [target_body_pos[0], target_body_pos[1], target_body_orn[2]]

        print(f"\nMoving Body {i}")
        
        # print("target_pose: ", target_pose)
        actions = controller._navigate_to_pose_direct(target_pose)
        # actions = controller._navigate_to_pose_direct((target_body_pos[0], target_body_pos[1], orientation))
        steps = 0
        for action in actions:
            action[5:12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Left arm and gripper
            action[12:19] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Right arm and gripper
            env.step(action)

        # # Getting delta arm positions
        # current_pos, current_orn = hand_positions[i], hand_rotations[i]
        # prev_pos, prev_orn = hand_positions[i - frame_step], hand_rotations[i - frame_step]

        # delta_pos = np.subtract(current_pos, prev_pos)
        # # delta_pos = np.array([delta_pos[0], delta_pos[2], delta_pos[1]])
        # delta_pos = np.array([delta_pos[2], delta_pos[0], delta_pos[1]])

        # target_hand_pos = list(np.add(prev_hand_pos, delta_pos))
        # target_hand_orn = list(R.from_matrix(hand_rotations[i]).as_quat())

        
        # # Getting debug box
        # # clear_debug_drawing()
        # # robot_position, robot_orientation = robot.get_position_orientation()
        # # Let's say the robot reaches the target body pos and ori (To remove the robot control part from this debugging)
        # robot_position, robot_orientation = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()
        
        # robot_to_world = np.eye(4)
        # robot_to_world[:3, :3] = R.from_quat(robot_orientation).as_matrix()
        # robot_to_world[:3, 3] = np.transpose(robot_position)
        # hand_position_robot_frame = np.array([target_hand_pos[0], target_hand_pos[1], target_hand_pos[2], 1])
        # hand_position_robot_frame = np.transpose(hand_position_robot_frame)
        # target_hand_pos_world_frame = np.matmul(robot_to_world, hand_position_robot_frame)
        # draw_box(center=(target_hand_pos_world_frame[0], target_hand_pos_world_frame[1], 
        #                  target_hand_pos_world_frame[2]), extents=(0.005, 0.005, 0.005), size=3)
        # prev_hand_position_robot_frame = np.array([prev_hand_pos[0], prev_hand_pos[1], prev_hand_pos[2], 1])
        # prev_hand_position_robot_frame = np.transpose(prev_hand_position_robot_frame)
        # prev_hand_pos_world_frame = np.matmul(robot_to_world, prev_hand_position_robot_frame)
        # draw_line(prev_hand_pos_world_frame[:3], target_hand_pos_world_frame[:3], size=3)
        # for _ in range(100):
        #     og.sim.step()

        
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
        
        if save_images:
            color_img = og.sim.viewer_camera._get_obs()[0]['rgb']
            cv2.imwrite(os.path.join(save_image_dir, f"{counter}.jpg"), color_img)
            counter += 1
            
        # change prev to current
        if arnav_method:
            prev_body_pos, prev_body_orn = target_body_pos, R.from_euler("xyz", target_body_orn).as_quat()
            # prev_hand_pos, prev_hand_orn = target_hand_pos, target_hand_orn

    
    if save_images:
        output_video = f"{video_name}.avi"
        create_video_from_images(image_folder=save_image_dir, output_video=output_video, fps=fps)

    
    for _ in range(10000):
            og.sim.step()
    
    og.shutdown()
            
        

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
