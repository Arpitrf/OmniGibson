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

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def convert_delta_to_absolute(action, robot):
    current_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
    current_pos = current_pose[0]
    current_orn = current_pose[1]
    # target_pos = 


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
    # Reset environment and robot
    env.reset()
    robot.reset()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # set head pos
    for _ in range(5):
        action = np.zeros(22)
        action[4] = -0.1
        action = action.tolist()
        env.step(action=action)
    
    # # close gripper
    # for _ in range(100):
    #     action = np.zeros(22)
    #     action[11] = -0.02
    #     action[12] = -0.02
    #     action = action.tolist()
    #     env.step(action=action)


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
    # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "size": 0.035,
            # "position": [-0.3, -0.8, 0.5],
            "position": [-0.5, -0.7, 0.5],
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

    # config['robots'][0]['orientation'] = desired_ori
    desired_pos = np.asarray((0.8,0.6,0))
    # config['robots'][0]['position'] = desired_pos
    # config['robots'][0]['default_reset_mode'] = 'tuck'

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    
    # Action primitives
    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # remove later
    # og.sim.restore('sim_state_block_pick3.json')
    # time.sleep(5)
    # scene.load_state

    # debug
    # print("defaut joint pos: ", robot._default_joint_pos, len(robot._default_joint_pos))
    # obs, _, _, _ = env.step(action=action)

    # # Allow user to move camera more easily
    # og.sim.enable_viewer_camera_teleoperation()


    # initializations for saving to file
    save_path = 'output_assisted'
    os.makedirs(f'{save_path}', exist_ok=True)
    # # Define the video codec, frame rate, and output video file
    # video_codec_mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'H264' codec for MP4 files
    # video_codec_avi = cv2.VideoWriter_fourcc(*'DIVX')
    # fps = 5  # Adjust the frame rate as needed

    # Load action and repeat it
    with open(f'{save_path}/goal_traj.pickle', 'rb') as f:
        teleop_traj = pickle.load(f)

    # # Record various noisy trajs and save the obs and traj
    # for i in range(1):
    #     i = 'goal_traj'
    #     custom_reset(env, robot)
    #     os.makedirs(f'{save_path}/{i}/rgb', exist_ok=True)
    #     os.makedirs(f'{save_path}/{i}/seg', exist_ok=True)
    #     os.makedirs(f'{save_path}/{i}/obs', exist_ok=True)

    #     new_traj = []
    #     traj_info = {}
    #     for step, action in enumerate(teleop_traj):
    #         # # add noise when ee pos is being controlled
    #         # if action[5] != 0.0 or action[6] != 0.0 or action[7] != 0.0:
    #         #     noise_pos = np.random.uniform(-1.0, 1.0, 3)
    #         #     noise_orn = np.random.uniform(-2.5, 2.5, 3)
    #         #     action[5:8] += noise_pos
    #         #     action[8:11] += noise_orn

    #         new_traj.append(action)
    #         obs, _, _, _ = env.step(action=action)
    #         # print("obs:")
    #         # print_dict_keys(obs)
            
    #         # save obs to file
    #         if args.save_traj:
    #             rgb = obs['robot0']['robot0:eyes:Camera:0']['rgb']
    #             rgb_pil = Image.fromarray(rgb[:,:,:3])
    #             rgb_pil.save(f"{save_path}/{i}/rgb/{step:04d}.jpg")
    #             seg_sem = obs['robot0']['robot0:eyes:Camera:0']['seg_semantic']
    #             plt.imshow(seg_sem)
    #             plt.savefig(f"{save_path}/{i}/seg/{step:04d}.jpg")
    #             with open(f"{save_path}/{i}/obs/{step:04d}.pickle", 'wb') as f:
    #                 pickle.dump(obs, f)

    #         # # covert rgb to video and save it as well
    #         # video_mp4.write(rgb[:,:,:3])
    #         # video_avi.write(rgb[:,:,:3])

    #         # fig, ax = plt.subplots(1,2)
    #         # ax[0].imshow(rgb)
    #         # ax[1].imshow(seg_sem)
    #         # plt.show()
    #     if args.save_traj:
    #         obj_in_hand = robot._ag_obj_in_hand['left']
    #         traj_info['obj_in_hand'] = obj_in_hand.name if obj_in_hand is not None else obj_in_hand
    #         traj_info['traj'] = new_traj
    #         with open(f"{save_path}/{i}/traj_info.pickle", 'wb') as f:
    #             pickle.dump(traj_info, f)

    # Teleop robot
    # custom_reset(env, robot)
    
    # add orientations and base mvmt
    action_keys = ['P', 'SEMICOLON', 'RIGHT_BRACKET', 'LEFT_BRACKET', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'Y', 'B', 'N', 'O', 'U', 'C', 'V', 'T']
    teleop_traj = []
    
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    og.sim.restore('dynamics_model_data/episode_00000_start.json')
    state = og.sim.dump_state()
    og.sim.stop()
    # trying increasing friction in fingers
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=400.0,
        dynamic_friction=400.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        print("arm, links: ", arm, links)
        for link in links:
            # print("link: ", link)
            for msh in link.collision_meshes.values():
                print("msh: ", msh)
                msh.apply_physics_material(gripper_mat)

    for obj in scene.objects:
        if obj.name == 'box':
            obj_box = obj
            from omni.isaac.core.materials import PhysicsMaterial
            primitive_mat = PhysicsMaterial(
                prim_path=f"{obj_box.prim_path}/box_mat",
                name="box_material",
                static_friction=400.0,
                dynamic_friction=400.0,
                restitution=None,
            )
            print("obj mass: ", obj.mass)
            for link in obj_box.links.values():
                # print("link: ", link)
                for msh in link.collision_meshes.values():
                    print("msh: ", msh)
                    msh.apply_physics_material(primitive_mat)
    
    og.sim.play()
    og.sim.load_state(state)
    # env.reset()
    # robot.reset()

    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    # # remove later
    # current_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
    # temp_pos = current_pose[0]

    max_steps = -1
    step = 0
    while step != max_steps:
        action, keypress_str = action_generator.get_teleop_action()
        # action = convert_delta_to_absolute(action, robot)
        # remove later
        # action[11] = -1
        # print("keypress_str: ", step, keypress_str)
        # print("action: ", action, len(action))

        # # set fixed orn
        # # fixed_orn = R.from_quat([0.50594286, -0.49429841, -0.49804673, -0.50163767]).as_euler('YXZ')
        # # fixed_orn = np.array(fixed_orn)
        # fixed_orn_rotvec = R.from_quat([0.50594286, -0.49429841, -0.49804673, -0.50163767]).as_rotvec()
        # fixed_orn = np.array(fixed_orn_rotvec)
        # # print("fixed_orn: ", fixed_orn)
        # action[8:11] = fixed_orn_rotvec
        # action[5:8] = np.array([0.70564498, 0.14813691, 0.68340246])
        
        # # # remove later
        # # # [0.50564498 0.24813691 0.78340246] [ 0.50594286 -0.49429841 -0.49804673 -0.50163767]
        # current_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
        # current_pos = current_pose[0]
        # current_orn = current_pose[1]
        # print("curr_pos, curr_orn: ", current_pos, current_orn)
        # temp = robot.get_relative_eef_pose()
        # print("temppp: ", temp)

        if keypress_str == 'TAB':
            temp = robot.get_relative_eef_pose(arm='right')
            print("robot eef pose: ", temp)
            temp = (robot.get_eef_position(), robot.get_eef_orientation())
            print("global robot eef pose: ", temp)
            break
        obs, _, _, _ = env.step(action=action)
        step += 1
        # print("keypress_str in action_keys: ", keypress_str, action_keys)
        if keypress_str in action_keys:
            # print("1111111111111111111111111111")
            print("action: ", action)
            teleop_traj.append(action)

        # remove later
        # print("collision?: ", detect_robot_collision_in_sim(robot))
        
        # if step % 50 == 0:
            # print("proprio_dict: ", robot._get_proprioception_dict().keys())
            # print("joint_qpos: ", robot._get_proprioception_dict()['joint_qpos'], robot._get_proprioception_dict()['joint_qpos'].shape)
            print("camera_qpos: ", robot._get_proprioception_dict()['camera_qpos'], robot._get_proprioception_dict()['camera_qpos'].shape)
    
    # # remove later
    # og.sim.save(f'sim_state_block_pick3.json')
    # arr = scene.dump_state(serialized=True)
    # with open('scene_state.pickle', 'wb') as f:
    #     pickle.dump(arr, f)
    
    # # Save traj to file
    # with open(f'{save_path}/goal_traj.pickle', 'wb') as f:
    #     pickle.dump(teleop_traj, f)

    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()