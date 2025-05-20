import os
import yaml
import  pdb
import pickle
import h5py
import imageio

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.objects import PrimitiveObject
from omnigibson.utils.python_utils import nums2array
from omnigibson.arpit_trial.move_base_vel import move_base_vel, detect_robot_collision_custom
from omnigibson.arpit_trial.utils.utils import hori_concatenate_image
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

def visualize_scan(env):
    obs, info = env.get_obs()
    robot_name = env.robots[0].name
    front_scan = obs[f'{robot_name}'][f'{robot_name}:base_front_laser_link:Lidar:0']['scan']
    rear_scan = obs[f'{robot_name}'][f'{robot_name}:base_rear_laser_link:Lidar:0']['scan']
    print("front_scan, rear_scan: ", front_scan.shape, rear_scan.shape)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(front_scan, color='blue', label='Front Scan')
    ax[1].plot(rear_scan, color='red', label='Rear Scan')
    ax[0].set_ylim(0, 2)
    ax[1].set_ylim(0, 2)
    plt.show()

def sample_delta_pose2d(threshold):
    delta_pose2d = []
    x_range = [-0.1, 0.5]
    y_range = [-0.4, 0.4]
    while True:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        norm = np.sqrt(x**2 + y**2)
        if norm > threshold:
            delta_pose2d.append(x)
            delta_pose2d.append(y)
            break
    yaw = np.random.uniform(-0.5, 0.5)
    delta_pose2d.append(yaw)
    return np.array(delta_pose2d)

def custom_reset(env, robot, episode_memory, episode_number):
    # scene_initial_state = env.scene._initial_state
    
    # scene.reset()
    
    # base_yaw = np.random.uniform(0, np.pi)
    # remove later
    base_yaw = np.pi / 2
    r_euler = R.from_euler('z', base_yaw) # or -120
    r_quat = R.as_quat(r_euler)
    robot.set_position_orientation(orientation=r_quat)
    
    # # every 100 episode we randomize the robot initial position
    # if episode_number % 1 == 0:
    #     while True:
    #         random_pos = scene.get_random_point(robot=robot)
    #         robot.set_position_orientation(position=random_pos[1])
    #         # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = random_pos
    #         base_collisions = []
    #         for _ in range(50): 
    #             robot_collision = detect_robot_collision_custom(robot) 
    #             base_collisions.append(robot_collision)
    #             og.sim.step()
    #         print("base_collision: ", sum(base_collisions))
    #         if sum(base_collisions) > 3:
    #             scene.reset()
    #         else:
    #             break

    for _ in range(50): og.sim.step()

def post_process_scan(scan, threshold=0.931): 
    """
    threshold is normailzed (it is equivalent to saying >1.6 meters)
    """
    # make any value greater than th 10.0
    scan[scan > threshold] = 10.0

    # make any self-collision values also 10.0
    scan[scan == 0.0] = 10.0

    # plt.plot(scan)
    # plt.ylim(-0.05, 1.6)
    # plt.show()

    return scan


def dump_to_memory(env, robot, episode_memory, base_collision):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    gripper_right_tool_link_pose = robot.get_relative_link_pose("gripper_right_tool_link")
    gripper_right_tool_link_pose = th.cat((gripper_right_tool_link_pose[0], gripper_right_tool_link_pose[1]))
    proprio['gripper_right_tool_link_pose'] = gripper_right_tool_link_pose
    arm_right_tool_link_pose = robot.get_relative_link_pose("arm_right_tool_link")
    arm_right_tool_link_pose = th.cat((arm_right_tool_link_pose[0], arm_right_tool_link_pose[1]))
    proprio['arm_right_tool_link_pose'] = arm_right_tool_link_pose
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    robot_name = env.robots[0].name
    # Add scan
    front_scan = obs[f'{robot_name}'][f'{robot_name}:base_front_laser_link:Lidar:0']['scan']
    rear_scan = obs[f'{robot_name}'][f'{robot_name}:base_rear_laser_link:Lidar:0']['scan']
    front_scan = front_scan.squeeze().numpy()
    rear_scan = rear_scan.squeeze().numpy()
    scan = np.concatenate((front_scan, rear_scan))
    scan = post_process_scan(scan)
    
    episode_memory.add_observation('scan', scan)
    
    episode_memory.add_extra('base_collision', base_collision)


def move_primitive(writer):
    base_collision = False
    episode_length = 8
    i = 0
    while not base_collision and i < episode_length:
        delta_pose2d = sample_delta_pose2d(threshold=0.2)
        print("detla_pose2d: ", delta_pose2d)
        
        # visualize_scan(env)
        # dump to memory the action
        episode_memory.add_action('actions', delta_pose2d)
        
        # execute action
        retval = move_base_vel(env, robot, action_primitives, delta_pose2d, writer)
        print("robot_collisions: ", retval["robot_base_collision"])
        
        # dump to memory the final state
        dump_to_memory(env, robot, episode_memory, base_collision=retval["robot_base_collision"])
        
        if retval["robot_base_collision"]:
            base_collision = True

        i += 1



# all_scenes = ["Rs_int", "Benevolence_2_int", "Benevolence_1_int", "Beechwood_0_int", "Ihlen_0_int", "Ihlen_1_int", "Merom_0_int", "Merom_1_int", "Pomaria_0_garden", "hall_conference_large"]
# all_scenes = ["Rs_int", "Benevolence_2_int", "Beechwood_0_int", "Ihlen_0_int", "Ihlen_1_int"]
# all_scenes = ["Merom_0_int", "Pomaria_0_int", "Wainscott_0_int", "Rs_int", "Ihlen_1_int"]
# remove later
# Done - Rs_int, Ihlen_0_int
# okis - Benevolence_2_int 
# good - Ihlen_1_int, Pomaria_0_int, Wainscott_0_int
all_scenes = ["Wainscott_0_int"]

config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["robots"][0]["controller_config"] = {
    "base": dict(),
    "arm_right": dict(),
    "arm_left": dict(),
    "gripper_right": dict(),
    "gripper_left": dict(),
}
config["robots"][0]["default_reset_mode"] = "tuck"
config["robots"][0]["default_trunk_offset"] = 0.30
config["robots"][0]["controller_config"]["base"]["name"] = "JointController"
config["robots"][0]["controller_config"]["base"]["motor_type"] = "velocity"
config["robots"][0]["controller_config"]["arm_right"]["name"] = "NullJointController"
config["robots"][0]["controller_config"]["arm_left"]["name"] = "NullJointController"
config["robots"][0]["controller_config"]["gripper_right"]["name"] = "NullJointController"
config["robots"][0]["controller_config"]["gripper_left"]["name"] = "NullJointController"

# Adding lidar 
config["robots"][0]["obs_modalities"].append("scan")
# config["robots"][0]["obs_modalities"].append("occupancy_grid")
config["robots"][0]["sensor_config"]["ScanSensor"] = {
        "modalities": ["scan"],  # if specified, this will override the values in robots_config["obs_modalities"]
        "enabled": True,
        "noise_type": None,
        "noise_kwargs": None,
        "sensor_kwargs": {
            # Lidar settings to closely mimic real Tiago's lidar readings
            "min_range": 0.15,
            "max_range": 1.6,
            "horizontal_fov": 240.0, # real tiago has 180 degrees, but if I use 180, there are big blind spots
            "vertical_fov": 1.0,
            "yaw_offset": 0.0,
            "horizontal_resolution": 0.44, # real tiago has 0.33 but since we changed fov and to keep the total values constant (545), we change this
            "vertical_resolution": 1.0,
            "rotation_rate": 0.0,
            "draw_points": False,
            "draw_lines": False
            # Occupancy Grid kwargs
            # "occupancy_grid_resolution": 128,
            # "occupancy_grid_range": 5.0,
            # "occupancy_grid_inner_radius": 0.5,
            # "occupancy_grid_local_link": None,
        }
    }

SAVE_VIDEO_FREQUENCY = 15
num_episodes_per_scene = 250
save_folder = 'base_collision_detection'
os.makedirs(save_folder, exist_ok=True)
episode_number = 0

for current_scene in all_scenes: 

    config["scene"] = dict()
    config["scene"]["type"] = "InteractiveTraversableScene"
    config["scene"]["scene_model"] = current_scene
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]

    if og.sim is not None:
        og.clear()
        og.sim.stop()
        # scene = InteractiveTraversableScene(scene_model=current_scene)
        # og.sim.import_scene(scene)
        env = og.Environment(configs=config)
        og.sim.play()
    else:
        env = og.Environment(configs=config)
        
    scene = env.scene
    robot = env.robots[0]

    for _ in range(20):
        og.sim.step()

    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    if os.path.isfile(f'{save_folder}/dataset.hdf5'):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)
    init_episode_number = episode_number

    # breakpoint()
    state = og.sim.dump_state(serialized=False)
    while episode_number < init_episode_number + num_episodes_per_scene:
        print(f"============== Episode {episode_number} ==============")
        writer = None
        episode_memory = Memory()

        if episode_number % SAVE_VIDEO_FREQUENCY == 0:
            imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
            output_path = f'{save_folder}/episode_{episode_number:05d}_video.mp4'
            writer = imageio.get_writer(output_path, **imgio_kargs)
        
        custom_reset(env, robot, episode_memory, episode_number)
        breakpoint()
        base_collision = False
        dump_to_memory(env, robot, episode_memory, base_collision=base_collision)
        move_primitive(writer)
        # episode_memory.dump(f'{save_folder}/dataset.hdf5')

        for _ in range(10):
            og.sim.step()
            if writer is not None:
                obs, _ = env.get_obs()
                img = obs[f"{robot.name}"][f"{robot.name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
                viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
                concat_img = hori_concatenate_image([viewer_img, img])
                concat_img = concat_img * 255.0
                concat_img = concat_img.astype(np.uint8)
                writer.append_data(concat_img)

        og.sim.load_state(state, serialized=False)
        for _ in range(10):
            og.sim.step()
        
        del episode_memory
        episode_number += 1
       
        # # =============================== Teleop ===============================
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

        # max_steps = -1 
        # step = 0
        # while step != max_steps:
        #     action, keypress_str = action_generator.get_teleop_action()
        #     # print("action: ", action)

            
        #     env.step(action=action)
        #     if keypress_str == 'TAB':
        #         robot_collision = detect_robot_collision_in_sim(robot)
        #         # print("Robot collision: ", robot_collision)
                
        #         breakpoint()
        #     step += 1
        # # ========================================================================


og.shutdown()