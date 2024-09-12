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
from omnigibson.objects import DatasetObject

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
        '/World/box/base_link/visuals',
        '/World/plate/base_link/visuals',
        '/World/apple/base_link/visuals',
        '/World/table/base_link/visuals'
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
    action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[12:19])))   
    episode_memory.add_action('actions', action_to_add)

    # 4. Move to a random pose in a neighbourhood
    # temp_pose = (org_pos + np.array([0.0, 0.0, 0.2]), org_quat)
    x = np.random.uniform(org_pos[0] - 0.1, org_pos[0] + 0.1)
    y = np.random.uniform(org_pos[1] - 0.1, org_pos[1] + 0.1)
    z = np.random.uniform(org_pos[2] + 0.2, org_pos[2] + 0.3)
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
    episode_memory.add_action('actions', np.zeros(10, dtype=np.float64))

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
    base_x_noise = np.random.uniform(-0.05, 0.05)
    base_y_noise = np.random.uniform(-0.05, 0.05)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = base_pos
    
    base_yaw = -120
    base_yaw_noise = np.random.uniform(-5, 5)
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

def navigate_primitive(action_primitives, env, robot, episode_memory):
    obj = env.scene.object_registry("name", "box")
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
    
    # target_pos, target_orn_quat = np.array([-4.9999990e-02, -4.0000005e-01, -1.8534674e-07]), np.array([-5.8735580e-08,  2.1243411e-08, -7.6604444e-01,  6.4278764e-01])
    target_pos, target_orn_quat = np.array([-4.9999990e-02, -4.0000005e-01, -1.8534674e-07]), np.array([ 0., 0., -0.8660254, 0.5])
    target_orn_euler = R.from_quat(target_orn_quat).as_euler('XYZ')
    pose2d = np.array([target_pos[0], target_pos[1], target_orn_euler[2]])
    # add some noise:
    noise_orn = np.random.uniform(-0.35, 0.35)
    noise_pos = np.random.uniform(-0.1, 0.1, 2)
    # # target_orn_euler[2] -= 0.35
    # target_pos[0] -= 0.1
    # pose2d += np.array([noise_pos[0], noise_pos[1], noise_orn])    
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(pose2d, episode_memory=episode_memory),
                         env,
                         robot,
                         gripper_closed,
                         episode_memory,
                         step,
                         grasp_change_inds)
    
    # # Adding all 0 action for the last step
    # episode_memory.add_action('actions', np.zeros(10, dtype=np.float64))
    
def main():
    # Initializations
    np.random.seed(11)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    
    rot = np.array(R.from_euler('z', 90, degrees=True).as_quat())
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
            "category": "shelf",
            "model": "lbbims",
            "scale": [3.0, 1.5, 1.5],
            "position": [-1.0, -0.7, 0.9],
            "orientation": rot
        }
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Set friction
    state = og.sim.dump_state()
    og.sim.stop()
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=20.0,
        dynamic_friction=20.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)
    og.sim.play()
    og.sim.load_state(state)

    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # save_folder = 'dynamics_model_dataset'
    save_folder = 'navigate_pick_prior'
    os.makedirs(save_folder, exist_ok=True)

    # Obtain the number of episodes
    episode_number = 0
    if os.path.isfile(f'{save_folder}/dataset.hdf5'):
        with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
            with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
                episode_number = len(file['data'].keys())
                print("episode_number: ", episode_number)

    
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

    
    for i in range(20):
        print(f"---------------- Episode {i} ------------------")
        start_time = time.time()
        episode_memory = Memory()

        # input("Press enter to reset")
        custom_reset(env, robot, episode_memory)

        # remove later
        box_obj = env.scene.object_registry("name", "box")
        filter_objs = [box_obj]
        
        # Loop control until user quits
        max_steps = -1 
        step = 0
        while step != max_steps:
            action, _ = action_generator.get_teleop_action()
            env.step(action=action)
            is_collision = detect_robot_collision_in_sim(robot, filter_objs=filter_objs, ignore_obj_in_hand=True)
            print("is_collision: ", step, is_collision)     
            if step % 50 == 0:
                proprio = robot._get_proprioception_dict()
                print(proprio['arm_right_qpos'])
            step += 1
        
        # ensure object is in initial view
        obs, obs_info = env.get_obs()
        seg_semantic = obs_info['robot0']['robot0:eyes:Camera:0']['seg_semantic']
        # print("seg_semantic.values(): ", seg_semantic.values())
        while 'object' not in seg_semantic.values():
            episode_memory = Memory()
            custom_reset(env, robot, episode_memory)
            obs, obs_info = env.get_obs()
            seg_semantic = obs_info['robot0']['robot0:eyes:Camera:0']['seg_semantic']

        for _ in range(10):
            og.sim.step()

        # base_pos, base_orn = robot.get_position_orientation()
        # print("base_pos, base_orn: ", base_pos, base_orn)
        # euler = R.from_quat(base_orn).as_euler('XYZ', degrees=True)
        # rotvec = R.from_quat(base_orn).as_rotvec()
        # print("euler: ", euler)
        # print("rotvec: ", rotvec / np.linalg.norm(rotvec), np.linalg.norm(rotvec))

        # # save the start simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_start.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
        #     pickle.dump(arr, f)
        
        # navigate_primitive(action_primitives, env, robot, episode_memory)
        # grasp_primitive(action_primitives, env, robot, episode_memory)
        # episode_memory.dump(f'{save_folder}/dataset.hdf5')

        # # save the end simulator state
        # og.sim.save(f'{save_folder}/episode_{episode_number:05d}_end.json')
        # arr = scene.dump_state(serialized=True)
        # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
        #     pickle.dump(arr, f)

        # # save video of the episode
        # save_video(np.array(episode_memory.data['observations']['rgb']), save_folder)

        for _ in range(50):
            og.sim.step()

        del episode_memory
        episode_number += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Episode {episode_number}: execution time: {elapsed_time:.2f} seconds")
            
        

if __name__ == "__main__":
    main()
