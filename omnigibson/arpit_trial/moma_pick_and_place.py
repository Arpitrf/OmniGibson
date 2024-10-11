import os
import yaml
import  pdb
import pickle
import h5py

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory

# def get_pose_wrt_robot():
#      # obtain target pose w.r.t robot
#     target_pose = np.eye(4)
#     target_pose[:3, :3] = R.from_quat(place_pose[1]).as_matrix()
#     target_pose[:3, 3] = np.transpose(place_pose[0])
#     robot_pose = robot.get_position_orientation()
#     robot_to_world = np.eye(4)
#     robot_to_world[:3, :3] = R.from_quat(robot_pose[1].numpy()).as_matrix()
#     robot_to_world[:3, 3] = np.transpose(robot_pose[0].numpy())

#     target_pose_wrt_robot = np.dot(np.linalg.inv(robot_to_world), target_pose) 
    
#     target_pos = target_pose_wrt_robot[:3, 3]
#     target_orn = np.array(R.from_matrix(target_pose_wrt_robot[:3, :3]).as_quat())
#     target_pose = (th.from_numpy(target_pos), th.from_numpy(target_orn))
#     print("target_pos: ", target_pos, target_orn)

def dump_to_memory(env, robot, episode_memory):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    for k in obs['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs['robot0']['robot0:eyes:Camera:0'][k].cpu().numpy())
    # # add gripper+object seg
    # gripper_obj_seg = obtain_gripper_obj_seg(obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'], obs_info['robot0']['robot0:eyes:Camera:0']['seg_instance_id'])
    # episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info['robot0']['robot0:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info['robot0']['robot0:eyes:Camera:0'][k])


    is_grasping = robot.custom_is_grasping()
    is_contact = detect_robot_collision_in_sim(robot)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_contact)

def custom_reset(env, robot, episode_memory): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

    # remove later
    base_pos = np.array([-0.05, -0.4, 0.0])
    base_x_noise = np.random.uniform(-0.15, 0.15)
    base_y_noise = np.random.uniform(-0.15, 0.15)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
    scene_initial_state['object_registry']['robot0']['root_link']['pos'] = base_pos
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset(head_joints_pos=head_joints)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    # add to memory
    dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, gripper_closed, episode_memory=None):
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            continue 
        if gripper_closed:
            # if left hand is ik
            # action[18] = -1
            # if left hand is joint controller
            action[20] = -1
        else: 
            # action[18] = 1
            action[20] = 1
        # print("action: ", action[14:17])
        env.step(action)
        # input("----")
        # for i in range(1):
        #     og.sim.step()
        #     input(f"completed {i} step")

        # debugging:
        # current_pos_world = robot.eef_links["right"].get_position_orientation()
        # print("current_pose_world: ", current_pos_world[0])


def primitive(episode_memory):
    gripper_closed = False

    # ======================= Move base ================================  
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.0, 0.0, 1.57])
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       gripper_closed, 
                       episode_memory)    
    # for _ in range(50):
    #     og.sim.step()
    curr_base_pos = robot.get_position()
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    # =================================================================================
    
    # action = th.zeros(robot.action_dim)
    # action[:2] = th.tensor([-0.1, 0.0])
    # env.step(action)
    # for _ in range(200):
    #     og.sim.step()

    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # w.r.t world
    target_pose = (th.tensor([0.1829, 0.4876, 0.4051]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    # w.r.t robot
    # target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), th.tensor([-0.0256,  0.0228,  0.6444,  0.7640]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       gripper_closed) 
    for _ in range(40):
        og.sim.step()
    current_pose_world = robot.eef_links["right"].get_position_orientation()
    # print("move hand down completed. Desired and Reached right eef pose reached: ", target_pose[0], current_pose_world[0])
    # =================================================================================

    # ============= Perform grasp ===================
    gripper_closed = True
    action = action_primitives._empty_action()
    # if left hand is IK
    # action[18] = -1
    # if left has is joint controller
    action[20] = -1
    execute_controller([action], env, robot, gripper_closed, episode_memory)
    # step the simulator a few steps to let the gripper close completely
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    dump_to_memory(env, robot, episode_memory)
    action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    episode_memory.add_action('actions', action_to_add)
    # ==============================================
        
    # ======================= Move hand up ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.1])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, in_world_frame=False, ignore_failure=True, gripper_closed=gripper_closed, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       gripper_closed,
                       episode_memory)
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
    #                    env, 
    #                    robot, 
    #                    gripper_closed)
    

    for _ in range(50):
        og.sim.step()
    
    # Debugging
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # input()
    # =================================================================================

    # ============= Move base ===================
    # debugging
    ee_pose_before_nav = robot.get_relative_eef_pose(arm='right')
    
    gripper_closed = True
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.456, 0.0257, 0.0]) # [0.526, 0.0257, 0.0]
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       gripper_closed, 
                       episode_memory)    
    # for _ in range(50):
    #     og.sim.step()
    curr_base_pos = robot.get_position()
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    
    # Debugging
    ee_pose_after_nav = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(ee_pose_after_nav[0] - ee_pose_before_nav[0])
    orn_error = T.get_orientation_diff_in_radian(ee_pose_after_nav[1], ee_pose_before_nav[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # input()
    # ============================================

    # action = th.zeros(robot.action_dim)
    # action[:2] = th.tensor([0.1, 0.0])
    # env.step(action)
    # for _ in range(200):
    #     og.sim.step()

    # # # remove later
    # # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_before_place.json'])

    # ======================= Move hand to place pose ================================
    # w.r.t world
    place_pose =  (th.tensor([ 1.0702, -0.1873,  0.9063]), th.tensor([-0.0488, -0.0116,  0.5546,  0.8306])) # [ 1.1602, -0.1873,  0.8463]
    # w.r.t robot
    # place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))
    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory, gripper_closed=gripper_closed), 
                       env, 
                       robot, 
                       gripper_closed,
                       episode_memory)
    # execute_controller(action_primitives._move_hand_direct_ik(place_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    gripper_closed)
    current_pose_world = robot.eef_links["right"].get_position_orientation()
    print("move hand to place location completed. Desired and Reached right eef pose reached: ", place_pose[0], current_pose_world)
    # input()
    # ====================================================================================

    # ============= Open grasp =================
    gripper_closed = False
    action = action_primitives._empty_action()
    # if left hand is IK
    # action[18] = -1
    # if left has is joint controller
    action[20] = 1
    execute_controller([action], env, robot, gripper_closed, episode_memory)
    # step the simulator a few steps to let the gripper close completely
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    dump_to_memory(env, robot, episode_memory)
    action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    episode_memory.add_action('actions', action_to_add)
    # ==========================================

    for _ in range(50):
        og.sim.step()



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_absolute_ori'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

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
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
    # {
    #     "type": "DatasetObject",
    #     "name": "box_of_baking_powder",
    #     "category": "box_of_baking_powder",
    #     "model": "vzgrlv",
    #     "mass": 1e-6,
    #     "position": [0.1, 0.5, 0.5],
    #     "orientation": box_quat
    # },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
print(robot.name)
# pdb.set_trace()

episode_memory = Memory()

# scene_initial_state = env.scene._initial_state
# base_yaw = 90
# r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
# r_quat = R.as_quat(r_euler)
# scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

# robot.set_orientation(r_quat)

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=100.0,
    dynamic_friction=100.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)

og.sim.play()
og.sim.load_state(state)

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
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# pdb.set_trace()
# obj = env.scene.object_registry("name", "box")
# obj.root_link.mass = 1e-2
# print("obj.mass: ", obj.mass)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)


for _ in range(50):
        og.sim.step()

save_folder = 'moma_pick_and_place'
os.makedirs(save_folder, exist_ok=True)
# Obtain the number of episodes

episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with FileLock(f'{save_folder}/dataset.hdf5' + ".lock"):
        with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
            episode_number = len(file['data'].keys())
            print("episode_number: ", episode_number)


for _ in range(1):
    custom_reset(env, robot, episode_memory)

    # save the start simulator state
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    arr = scene.dump_state(serialized=True)
    with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
        pickle.dump(arr, f)

    primitive(episode_memory)

    episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # save the end simulator state
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    arr = scene.dump_state(serialized=True)
    with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
        pickle.dump(arr, f)

# # Teleop
# max_steps = -1 
# step = 0
# # # pdb.set_trace()
# while step != max_steps:
#     action, keypress_str = action_generator.get_teleop_action()
#     # # remove later
#     # action[20] = -1
#     env.step(action=action)
#     if keypress_str == 'TAB':
#         right_eef_pose = robot.get_relative_eef_pose(arm='right')
#         right_eef_pose_world = robot.eef_links["right"].get_position_orientation()
#         base_pose = robot.get_position_orientation()
#         print("right_eef_pose: ", right_eef_pose)
#         print("right_eef_pose_world: ", right_eef_pose_world)
#         print("base_pose: ", base_pose)
#     step += 1

for _ in range(500):
    og.sim.step()

# Always shut down the environment cleanly at the end
og.clear()



