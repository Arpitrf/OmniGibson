import os
import yaml
import  pdb
import pickle
import cv2

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from memory import Memory
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson import object_states


# temp_prior = th.tensor([
#     [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.,   ],
#     [ 0.   ,  0.,     0.,     0.,     0.,     0.2,   -0.,    -0.,    -0.,     -1.,   ],
#     [ 0.   ,  0.,     0.,     0.007, -0.001,  0.207, -0.028, -0.105, -0.015,  -1.,   ],
#     # - Rotation -
#     # [ 0.   ,  0.,    -1.522,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
#     [ 0.   ,  0.,    -1.922,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
#     # ------
#     [ 0.527,  0.026,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
#     [ 0.   ,  0.,    -0.089,  0.,     0.,     0.,     0.,     0.,     0.,     -1.,   ],
#     # -- Move hand front --
#     # [ 0.   ,  0.,     0.,     0.191, -0.052,  0.028,  0.029, -0.384, -0.44,   -1.,   ],
#     [ 0.   ,  0.,     0.,     0.101, -0.052,  0.128,  0.029, -0.384, -0.44,   -1.,   ],
#     # -----
#     [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.,   ]
# ])

temp_prior = th.tensor([
    [ 0.   ,  0.,    -0.301,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.104,  0.334,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.267,  0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.1,   -0.,    -0.,    -0.,    -1.   ],
    [ 0.   ,  0.,    -1.456,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.459,  0.034,  0.,     0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,    -0.109,  0.,     0.,     0.,     0.,     0.,     0.,    -1.   ],
    [ 0.   ,  0.,     0.,     0.035, -0.016,  0.135, -0.015, -0.103, -0.065, -1.   ],
    [ 0.   ,  0.,     0.,     0.044, -0.017,  0.142, -0.016, -0.095, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.038, -0.017,  0.144, -0.016, -0.103, -0.067, -1.   ],
    [ 0.   ,  0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     1.   ],
])

def get_episode_reward(robot):
    target_pos = [1.1888, -0.1884,  0.8387]
    curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
    dist = np.linalg.norm(target_pos - curr_pos)
    return -dist

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return concatenated_image

def write_moviepy_video(obs_list, name, folder_path, fps=1):
    # print("pathhhhh: ", name)
    obs_list = np.array(obs_list)
    if isinstance(obs_list, np.floating) or obs_list.dtype == "float" or obs_list.dtype == "float32":
        obs_list = (obs_list * 255).astype(np.int16)
    obs_list = [np.int16(element) for element in obs_list]

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(obs_list, fps=fps)
    if not name.endswith(".mp4"):
        name = f"{folder_path}/{name}.mp4"
    clip.write_videofile(f"{name}", fps=fps, logger=None)

def add_noise(temp_prior):
    temp_prior_modified = temp_prior.clone()
    traj_length = len(temp_prior_modified)
    x_noise = np.random.multivariate_normal(mu_x, sigma_x)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z)
    yaw_noise = np.random.multivariate_normal(mu_yaw, sigma_yaw)
    print("yaw_noise: ", yaw_noise)
    # episode noise for x,y,z
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)
    episode_yaw_noise = yaw_noise

    counter = 0
    for i in range(1, traj_length-1):
        if i in [1, 2, 6]:
            # add noise to arm
            noise = th.tensor([x_noise[counter], y_noise[counter], z_noise[counter]])
            temp_prior_modified[i, 3:6] += noise
            counter += 1
        elif i in [3]:
            # add noise to base
            # noise_base_xy = np.random.uniform(-0.1, 0.1, 2)
            noise_base_xy = [0.0, 0.0]
            # noise_base_yaw = np.random.uniform(-0.3, 0.3)
            noise_base_yaw = yaw_noise[0]
            # pdb.set_trace()
            temp_prior_modified[i, 0:3] += th.tensor([noise_base_xy[0], noise_base_xy[1], noise_base_yaw])
    return temp_prior_modified, episode_pos_noise, episode_yaw_noise

def move_primitive(action):
    current_pose = robot.get_relative_eef_pose(arm='right')
    current_pos = current_pose[0]
    current_orn = current_pose[1]
    
    # print("action: ", action)
    delta_pos = action[3:6]
    # print("delta_pos: ", delta_pos)
    delta_orn = action[6:9]
    grasp_action = action[9]
    # print("grasp_action: ", grasp_action)
    
    target_pos = current_pos + delta_pos
    # print("current_pos, target_pos: ", current_pos, target_pos)
    # print("type(target_pos): ", type(target_pos))
    target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
    # print("target_orn: ", target_orn, target_orn.as_quat())
    target_orn = th.tensor(target_orn.as_quat())

    target_pose = (target_pos, target_orn)
    # print("current_pose: ", current_pose)
    print("target_pose: ", target_pose)
    print("current_joint_pos: ", robot.get_joint_positions()[robot.arm_control_idx["right"]])

    obs, info, total_collisions1 = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                            stop_on_contact=False,
                                                                            ignore_failure=True,
                                                                            stop_if_stuck=False), grasp_action)
    

    # obtain target pose2d
    current_base_pos, current_base_orn_quat = robot.get_position_orientation()
    current_base_yaw = R.from_quat(current_base_orn_quat).as_euler('XYZ')[2]

    # print("action: ", action)
    delta_base_pos = action[0:2] # this is in the robot frame
    # conver delta pos from robot frame to world frame
    robot_to_world = np.eye(4)
    robot_to_world[:3, :3] = R.from_quat(current_base_orn_quat).as_matrix()
    robot_to_world[:3, 3] = np.transpose(np.array([0.0, 0.0, 0.0]))
    delta_base_pos_homo = np.array([delta_base_pos[0], delta_base_pos[1], 0.0, 1.0])
    delta_base_pos_world = np.dot(robot_to_world, delta_base_pos_homo)
    delta_base_pos_world = th.from_numpy(delta_base_pos_world)
    delta_base_yaw = action[2]
    # # remove later
    # delta_base_yaw = 0.78

    # target_base_pos = current_base_pos + delta_base_pos_world[:3]
    target_base_pos = current_base_pos + th.tensor([delta_base_pos[0], delta_base_pos[1], 0.0])
    target_base_yaw = current_base_yaw + delta_base_yaw
    target_pose2d = th.tensor([target_base_pos[0], target_base_pos[1], target_base_yaw])
    # print("current_base_pos, delta_base_pos: ", current_base_pos[:2], delta_base_pos_world[:2])
    # print("current_base_yaw, delta_base_yaw: ", current_base_yaw, delta_base_yaw)
    obs, info, total_collisions2 = execute_controller(action_primitives._navigate_to_pose_direct(target_pose2d), grasp_action)


    # Hack to ensure that even if primitive does not return any action (if delta pose is 0), grasp action is performed
    action = action_primitives._empty_action()
    obs, info, total_collisions3 = execute_controller([action], grasp_action)

    # print("total_collisions1, total_collisions2, total_collisions3: ", total_collisions1, total_collisions2, total_collisions3)
    total_collisions = max(total_collisions1, total_collisions2, total_collisions3)

    for _ in range(40):
        og.sim.step()

    ee_pose_after = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # input()

    return obs, info, total_collisions

def execute_controller(ctrl_gen, grasp_action):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[20] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        box = env.scene.object_registry("name", "box")
        arm_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
        if arm_in_collision:
            total_collisions += 1
        # print("arm_in_collision, total_collisions: ", arm_in_collision, total_collisions)
    return obs, info, total_collisions

def perform_grasp():
    # ======================= Move hand to grasp pose ================================    
    grasp_action = 1.0
    # w.r.t world
    target_pose = (th.tensor([0.1829, 0.4876, 0.4051]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    # w.r.t robot
    # target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), th.tensor([-0.0256,  0.0228,  0.6444,  0.7640]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                       grasp_action) 
    for _ in range(40):
        og.sim.step()
    # current_pose_world = robot.eef_links["right"].get_position_orientation()
    # print("move hand down completed. Desired and Reached right eef pose reached: ", target_pose[0], current_pose_world[0])
    # =================================================================================

def correct_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=12.0,
        dynamic_friction=12.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset(head_joints_pos=head_joints)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=5)
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
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# og.sim.restore(["moma_pick_and_place/episode_00001_start.json"])
og.sim.restore(["moma_pick_and_place/temp.json"])

correct_gripper_friction()

shelf = env.scene.object_registry("name", "shelf")
shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

for _ in range(50):
    og.sim.step()


# box = env.scene.object_registry("name", "box")
# coffee_table = env.scene.object_registry("name", "coffee_table")
# shelf = env.scene.object_registry("name", "shelf")


# traj_length = len(temp_prior)
traj_length = 3 # for the place subtask
start_idx = 8
num_samples = 5
num_top_samples = 3
epochs = 10
success = False

mu_x = np.zeros(3)  # Example: 2-dimensional problem
sigma_x = np.eye(3) * 0.001
mu_y = np.zeros(3)  # Example: 2-dimensional problem
sigma_y = np.eye(3) * 0.001
mu_z = np.zeros(3)  # Example: 2-dimensional problem
sigma_z = np.eye(3) * 0.001


def sample_actions(t, actions):
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

def safe(action, use_hack=False):
    safe = True
    prev_state = og.sim.dump_state()
    box = env.scene.object_registry("name", "box")
    obj_in_hand_pos_before = box.get_position()
    
    if use_hack:
        total_collisions = 0
        print("Opening gripper")
        robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
        for _ in range(40):
            og.sim.step()
        # print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])
    else:
        _, _, total_collisions = move_primitive(action)
        # remove later
        # temp_state = og.sim.dump_state()
        # x = input("Press o to open gripper")
        # if x == 'o':
        #     robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
        #     for _ in range(40):
        #         og.sim.step()
        #     print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])
        #     og.sim.load_state(temp_state)
    
    obj_in_hand_pos_after = box.get_position()
    delta_pos_z = abs(obj_in_hand_pos_before[2] - obj_in_hand_pos_after[2]) 
    print("delta_pos_z: ", delta_pos_z)
    # print("total_collisions: ", total_collisions)

    normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
    # print("normalized_qpos: ", normalized_qpos)
    close_to_one = th.isclose(normalized_qpos[:-2], th.tensor(1.0), atol=1e-2)
    close_to_neg_one = th.isclose(normalized_qpos[:-2], th.tensor(-1.0), atol=1e-2)
    any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
    if any_close_to_one_or_neg_one:
        safe = False 

    # object dropped (unsafe)
    if delta_pos_z > 0.35:
        safe = False 
    # collisions
    if total_collisions > 0:
        safe = False
    
    if not safe:
        # Hack to make sure that load_state will work. I think there is an issue in using og.sim.load_state() when there are weird collisions
        robot.set_position_orientation(position=th.tensor([-2.0, 0.0, 0.0]))
        og.sim.load_state(prev_state)
        for _ in range(30):
            og.sim.step()
        # input("Reloaded state. Is it ok?")
    
    print("is this action safe? ", safe)
    # input()
    return safe

def check_success():
    box = env.scene.object_registry("name", "box")
    shelf = env.scene.object_registry("name", "shelf")
    obj_in_shelf = box.states[object_states.Inside].get_value(shelf)
    #TODO: Figure out why this doesn't work: robot._ag_obj_in_hand[robot.default_arm]
    grasping = robot.custom_is_grasping()
    success = obj_in_shelf and not grasping
    return success
    # return False

def undo_action(t, action):
    print("Undoing action")
    a = th.cat((-action[t][:-1], action[t][-1:])) 
    move_primitive(a)


def func(t, actions):
    
    if t == traj_length:
        print("Reached end of recursion")
        # open gripper and see
        a = th.zeros(10)
        a[-1] = 1.0
        # input("open gripper action")
        return safe(a, use_hack=True)
    
    for action in actions:
        print("--- time step, action: ", t, action[t][3:6])
        ee_pose_before = robot.get_relative_eef_pose(arm='right')
        joint_pos_before = robot.get_joint_positions()[robot.arm_control_idx["right"]]
        sim_state_before = og.sim.dump_state()
        if safe(action[t]):
            # In the current implementation I am performing the action (move_primitive) inside the safe action. This will change later.
            all_failed = func(t+1, actions)

            # if task success
            if check_success():
                print("Task succeeded!")
                all_failed = False 
                return all_failed
            
            # undo the last action. For now try making it go back to exact joint positions
            action = action_primitives._empty_action()
            action[robot.gripper_action_idx["right"]] = -1.0
            env.step(action, explicit_joints=joint_pos_before)
            for _ in range(40):
                og.sim.step()
            # undo_action(t, action)        
                    
            ee_pose_after = robot.get_relative_eef_pose(arm='right')
            pos_error = np.linalg.norm(ee_pose_after[0] - ee_pose_before[0])
            orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], ee_pose_before[1])
            print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
            print("joint_pos_before: ", joint_pos_before)
            joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            print("joint_pos after rewind: ", joint_pos_after)
            if any(abs(joint_pos_before - joint_pos_after) > 0.1):
                og.sim.load_state(sim_state_before)
                for _ in range(30):
                    og.sim.step()
                joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                print("joint_pos after reset: ", joint_pos_after)
            # input("Undid the action. Press enter to continue")

            # reset the shelf in case it has moved
            shelf = env.scene.object_registry("name", "shelf")
            print("BEFORE self pos, orn: ", shelf.get_position_orientation())
            shelf.set_position_orientation(shelf_pos_orn[0], shelf_pos_orn[1])
            for _ in range(10):
                og.sim.step()
            print("AFTER self pos, orn: ", shelf.get_position_orientation())

            if all_failed:
                # sample t+1 actions again
                actions = sample_actions(t+1, actions)
                # input("Resampled")

    all_failed = True
    return all_failed 

shelf = env.scene.object_registry("name", "shelf")
shelf_pos_orn = shelf.get_position_orientation()

# # replaying init traj
# traj_length = len(temp_prior)
# for t in range(start_idx, traj_length):
#     prev_state = og.sim.dump_state()
#     move_primitive(temp_prior[t])

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

# robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])

for _ in range(30):
    og.sim.step()

func(t=0, actions=actions)

# state = og.sim.dump_state()
# og.sim.save([f'/tmp/temp.json'])

# og.sim.stop()
# # Set friction
# from omni.isaac.core.materials import PhysicsMaterial
# gripper_mat = PhysicsMaterial(
#     prim_path=f"{robot.prim_path}/gripper_mat",
#     name="gripper_material",
#     static_friction=100.0,
#     dynamic_friction=100.0,
#     restitution=None,
# )
# for arm, links in robot.finger_links.items():
#     for link in links:
#         for msh in link.collision_meshes.values():
#             msh.apply_physics_material(gripper_mat)

# og.sim.play()
# og.sim.load_state(state)
# og.sim.restore(["/tmp/temp.json"])
# correct_gripper_friction()

# for _ in range(50):
#     og.sim.step()

# action = action_primitives._empty_action()
# for _ in range(20):
#     obs, info, total_collisions3 = execute_controller([action], grasp_action=1.0)
# input()

# place_pose =  (th.tensor([ 1.0702, -0.1873,  0.9063]), th.tensor([-0.0488, -0.0116,  0.5546,  0.8306])) 
# obs, info, total_collisions1 = execute_controller(action_primitives._move_hand_direct_ik(place_pose,
#                                                                             stop_on_contact=False,
#                                                                             ignore_failure=True,
#                                                                             stop_if_stuck=False,
#                                                                             in_world_frame=True), grasp_action=-1.0)
# robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])

# for _ in range(50):
#     og.sim.step()

# action = action_primitives._empty_action()
# for _ in range(20):
    # obs, info, total_collisions3 = execute_controller([action], grasp_action=1.0)
# input()

# # reset the shelf in case it has moved
# orn = shelf_pos_orn[1] + th.tensor([0.1, 0.2, 0.0, 0.8]) 
# orn = orn / th.linalg.norm(orn)
# shelf.set_position_orientation(shelf_pos_orn[0] + th.tensor([0.5, 0.0, 0.0]), orn)

# for i in range(30):
#     print(i)
#     og.sim.step()

# # reset the shelf in case it has moved
# shelf.set_position_orientation(shelf_pos_orn[0], shelf_pos_orn[1])

   

# traj_length = len(temp_prior)
# start_idx = 0
# for t in range(start_idx, traj_length):
#     prev_state = og.sim.dump_state()
#     move_primitive(temp_prior[t])
#     for _ in range(30):
#         og.sim.step()
#     if t == 2:
#         perform_grasp()

#     if t == 7:
#         og.sim.save([f'moma_pick_and_place/temp.json'])
#         break

    # checking go to previous state
    # if t == 9:
    #     og.sim.load_state(prev_state)
    #     for _ in range(500):
    #         og.sim.step()

    # # checking retract previous action
    # if t == 9:
    #     temp_prior_modified = th.cat((-temp_prior[t][:-1], temp_prior[t][-1:])) 
    #     move_primitive(temp_prior_modified)
    #     for _ in range(500):
    #         og.sim.step()

    # For placing action
    # 1. Sample 5 actions (5, 3)
    # 2. Check if ith action is safe. If safe execute ith action. Else try the next action. If all actions exhausted, retract 1 step
    # 2.1. To detect GT failures, try, if failure, go back to the previous state
        

    # Check all 5 actions
    # when go back t = t-1. resample the following actions and execute the new current action
    # if new current action is over, t = t - 1 again 
    # x_noise = np.random.multivariate_normal(mu_x, sigma_x, 5)
    # y_noise = np.random.multivariate_normal(mu_y, sigma_y, 5)
    # z_noise = np.random.multivariate_normal(mu_z, sigma_z, 5)
    # episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
    #                 np.expand_dims(y_noise, axis=1), 
    #                 np.expand_dims(z_noise, axis=1)), axis=1)

    # actions = actions[:, t:-1, 3:6] + episode_pos_noise[:, t:-1]
    # safe = True
    # if safe:
    #     #execute action
    #     func(t+1, actions)

    
    # detect failure
    # if failure:
        # 1. Undo the state (this is only needed in sim)
        # 2. Try other action

     



for _ in range(500):
    og.sim.step()

if success:
    print("Achieved success :)")
else:
    print("Failed :(")

og.shutdown()