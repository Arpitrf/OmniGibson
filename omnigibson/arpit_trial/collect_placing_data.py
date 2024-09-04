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
    # curr_right_arm_joints = np.array(proprio['arm_right_qpos'])
    default_joint_pos = robot.untucked_default_joint_pos[robot.arm_control_idx['right']]

    # print("proprio: ", proprio.keys())
    noise_1 = np.random.uniform(-0.2, 0.2, 3)
    # noise_2 = np.random.uniform(-0.1, 0.1, 4)
    noise_2 = np.random.uniform(-0.01, 0.01, 4)
    noise = np.concatenate((noise_1, noise_2))
    # print("arm_qpos.shape, noise.shape: ", curr_right_arm_joints.shape, noise.shape)
    right_hand_joints_pos = default_joint_pos + noise 
    # right_hand_joints_pos = default_joint_pos 

    scene_initial_state = env.scene._initial_state
    # for manipulation
    # scene_initial_state['object_registry']['robot0']['joints']['head_2_joint']['target_pos'] = np.array([-0.83])
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = [-0.05, -0.4, 0.0]
    r_euler = R.from_euler('z', -120, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

    head_joints_pos = np.array([-0.5031718015670776, -0.9972541332244873])
    # head_joints_pos = np.array([0.0, -0.83])

    # Reset environment and robot
    env.reset()
    robot.reset(right_hand_joints_pos=right_hand_joints_pos, head_joints_pos=head_joints_pos)
    # robot.reset()

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    # add to memory
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
        # print("action: ", action)
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

def grasp_primitive(action_primitives, env, robot, episode_memory):
    gripper_closed = False
    step = 0
    grasp_change_inds = []
    
    # Set grasp pose
    # global_pose3 = (np.array([-0.30722928, -0.79808354,  0.4792919 ]), np.array([-0.59315057,  0.37549363,  0.63922846,  0.3139489]))
    # right hand 1: (array([ 0.50431009, -0.25087801,  0.50123985]), array([ 0.57241185,  0.58268626, -0.41505368,  0.4006892 ]))
    org_pos, org_quat = np.array([ 0.48431009, -0.25087801,  0.46123985]), np.array([ 0.57241185,  0.58268626, -0.41505368,  0.4006892 ])

    # # get random pose in a cube
    # x_pos = np.random.uniform(0.3, 0.65)
    # y_pos = np.random.uniform(-0.45, 0.0)
    # z_pos = np.random.uniform(0.45, 0.75)
    # new_pos = np.array([x_pos, y_pos, z_pos])
    # # new_pos = np.array([0.4, -0.2, 0.7])


    # add noise to the position
    # new_pos = org_pos + np.random.uniform(-0.02, 0.02, 3)
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

    # If want to keep the original target pose
    new_pos, new_quat = org_pos, org_quat
    # new_quat = org_quat

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
    dump_to_memory(env, robot, episode_memory)    
    episode_memory.add_action('actions', action[12:19])

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

    # try placing
    for o in env.scene.objects:
        if o.name == 'plate':
            bowl_obj = o
    step, gripper_closed = execute_controller(action_primitives._place_on_top(bowl_obj),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    
    # Adding all 0 action for the last step
    episode_memory.add_action('actions', np.zeros(7, dtype=np.float64))

    
def main():
    # Initializations
    np.random.seed(5)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    scene_file = '/home/arpit/test_projects/OmniGibson/prior/episode_00012_end.json'
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "Cube"]
    config["scene"]['scene_file'] = scene_file
    
    # # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    temp_rot = np.array([0, 0, 0.9, 0.2])
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
            "name": "mixing_bowl", 
            "category": "mixing_bowl",
            "model": "deudkt", 
            "position": [-0.5, -0.95, 0.5],
            # "scale": [0.1, 0.05, 0.1],
        },
        # {
        #     "type": "DatasetObject",
        #     "name": "half_club_sandwich",
        #     "category": "half_club_sandwich",
        #     "model": "qkhepd",
        #     "position": [-0.5, -0.7, 0.45],
        #     # "orientation": [0, 0, 0, 1],
        #     "orientation": temp_rot / np.linalg.norm(temp_rot),
        #     "scale": [0.5, 0.5, 1]
        # },
        # {
        #     "type": "DatasetObject",
        #     "name": "plate",
        #     "category": "plate",
        #     "model": "pkkgzc",
        #     "position": [-0.5, -0.95, 0.5]
        # }    
        # {
        #     "type": "DatasetObject",
        #     "name": "table",
        #     "category": "breakfast_table",
        #     "model": "rjgmmy",
        #     "scale": [0.3, 0.3, 0.3],
        #     "position": [-0.7, 0.5, 0.2],
        #     "orientation": [0, 0, 0, 1]
        # }
    ]

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
    save_folder = 'prior'
    os.makedirs(save_folder, exist_ok=True)

    # og.sim.restore('/home/arpit/test_projects/OmniGibson/prior/episode_00012_end.json')
    # # step the simulator a few times 
    # print("RESTORING")
    # for _ in range(1000):
    #     og.sim.step()
    
    # with open('/home/arpit/test_projects/OmniGibson/prior/episode_00012_end.pickle', 'rb') as f:
    #     state = pickle.load(f)
    # og.sim.load_state(state)

    # # Obtain the number of episodes
    # episode_number = 0
    # if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    #     with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
    #         with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
    #             episode_number = len(file['data'].keys())
    #             print("episode_number: ", episode_number)

    # for _ in range(200):
    #         og.sim.step()

    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=10.0,
        dynamic_friction=10.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)
    og.sim.play()
    og.sim.load_state(state)

    # ------------------ teleop -------------------------
    episode_memory = Memory()
    custom_reset(env, robot, episode_memory)
    
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
        if keypress_str in action_keys:
            print("action: ", action)
            teleop_traj.append(action)
        if step % 100 == 0:
            print("robot right eef pose: ", robot.get_relative_eef_pose(arm='right'))
            # print("proprio_dict: ", robot._get_proprioception_dict().keys())
            # print("joint_qpos: ", robot._get_proprioception_dict()['joint_qpos'], robot._get_proprioception_dict()['joint_qpos'].shape)
            # print("camera_qpos: ", robot._get_proprioception_dict()['camera_qpos'], robot._get_proprioception_dict()['camera_qpos'].shape)
    # ---------------------------------------------------

    # for i in range(2):
    #     print(f"---------------- Episode {i} ------------------")
    #     start_time = time.time()
    #     episode_memory = Memory()

    #     custom_reset(env, robot, episode_memory)

    #     for _ in range(50):
    #         og.sim.step()

    #     # # save the start simulator state
    #     # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_start.json')
    #     # arr = scene.dump_state(serialized=True)
    #     # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
    #     #     pickle.dump(arr, f)
        
    #     grasp_primitive(action_primitives, env, robot, episode_memory)
    #     # episode_memory.dump(f'{save_folder}/dataset.hdf5')

    #     # # save the end simulator state
    #     # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_end.json')
    #     # arr = scene.dump_state(serialized=True)
    #     # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
    #     #     pickle.dump(arr, f)

    #     # # save video of the episode
    #     # save_video(np.array(episode_memory.data['observations']['rgb']), save_folder)

    #     # del episode_memory
    #     # episode_number += 1
        
    # #     # end_time = time.time()
    # #     # elapsed_time = end_time - start_time
    # #     # print(f"Episode {episode_number}: execution time: {elapsed_time:.2f} seconds")
            
        

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
