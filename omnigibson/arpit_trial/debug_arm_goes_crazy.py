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


def execute_controller(ctrl_gen, grasp_action):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[20] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)
    return obs, info, total_collisions


def correct_gripper_friction():
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


set_all_seeds(seed=2)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'absolute_pose'
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


grasp_action = -1

state = og.sim.dump_state()
for _ in range(10):
    robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = 1.0
    env.step(action)
    for _ in range(40):
        og.sim.step()
    print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])
    og.sim.load_state(state)
input()


state = og.sim.dump_state()
init_pose = robot.get_relative_eef_pose(arm='right')

# tensor([ 0.4959, -0.1955,  0.8050]), tensor([-0.3694, -0.1908,  0.5886,  0.6933]
# (tensor([ 0.5450, -0.1469,  0.9465]), tensor([-0.5662, -0.3046,  0.5108,  0.5707], dtype=torch.float64))

for _ in range(50):
    x = np.random.uniform(0.2, 0.6)
    y = np.random.uniform(-0.4, 0.1)
    z = np.random.uniform(0.3, 1.0)
    # target_pose = (th.tensor([x, y, z]), init_pose[1])
    # target_pose = (th.tensor([ 0.4959, -0.1955,  0.8050]), th.tensor([-0.3694, -0.1908,  0.5886,  0.6933]))
    # target_pose = (th.tensor([ 0.6804, -0.3934,  1.1684]), th.tensor([ 0.8787, -0.0379,  0.2288, -0.4171]))
    # print("target_pos: ", target_pose[0])
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose,
    #                                                             stop_on_contact=False,
    #                                                             ignore_failure=True,
    #                                                             stop_if_stuck=False,
    #                                                             in_world_frame=False), grasp_action)
    # target_joint_pos = th.tensor([1.0651,  0.8180,  0.3174,  1.6822, -0.2378, -1.4135,  2.0944])
    # target_joint_pos = th.tensor([ 0.7735,  0.6244,  0.8117,  1.7455,  0.2389, -1.1245,  1.8306])
    # target_joint_pos = th.tensor([ 0.7736,  0.6253,  0.8104,  1.7491,  0.2281, -1.1123,  2.0944])
    # target_joint_pos = th.tensor([ 0.8234,  0.7053,  0.7962,  1.4906,  0.3294, -1.2237,  2.0944])
    target_joint_pos = th.tensor([ 0.9648,  0.6953,  0.4068,  1.5236, -0.0171, -1.4137,  2.0944])

    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = -1.0
    env.step(action, explicit_joints=target_joint_pos)
    # robot.set_joint_positions(th.cat((target_joint_pos, th.tensor([0.0, 0.0]))), indices=th.cat((robot.arm_control_idx["right"], robot.gripper_control_idx["right"])))
    # robot.set_joint_positions(target_joint_pos, indices=robot.arm_control_idx["right"])
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = -1.0
    # env.step(action)
    # for _ in range(40):
    #     og.sim.step()
    
    for _ in range(200):
        og.sim.step()
    print("joint_pos after: ", robot.get_joint_positions()[robot.arm_control_idx["right"]])
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    # pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    # orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

    print("Next pose")
    # target_pose = (th.tensor([ 0.6104, -0.3034,  1.0684]), th.tensor([ 0.8787, -0.0379,  0.2288, -0.4171]))
    # target_pose = (th.tensor([ 0.6340, -0.2835,  0.8501]), th.tensor([-0.4861, -0.0453,  0.2437,  0.8380]))
    # target_pose = (th.tensor([ 0.5775, -0.1742,  1.0174]), th.tensor([-0.3370, -0.1598,  0.5262,  0.7642]))
    # target_pose = (th.tensor([ 0.4839, -0.1678,  0.9654]), th.tensor([-0.2359, -0.0928,  0.5362,  0.8051]))
    # target_pose = (th.tensor([ 0.5671, -0.1693,  1.0003]), th.tensor([-0.0929, -0.0053,  0.5755,  0.8125]))
    target_pose = (th.tensor([ 0.6182, -0.3037,  0.9785]), th.tensor([-0.2119, -0.0226,  0.4928,  0.8437]))
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                stop_on_contact=False,
                                                                ignore_failure=True,
                                                                stop_if_stuck=False,
                                                                in_world_frame=False), grasp_action)
    
    # # abs
    # action = th.zeros(robot.action_dim)
    # orn = R.from_quat(target_pose[1]).as_rotvec()
    # action[14:17] = th.tensor(target_pose[0])
    # action[17:20] = th.tensor(orn)
    # action[20] = -1
    # env.step(action)

    for i in range(200):
        print(i, robot._controllers["arm_right"]._goal)
        og.sim.step()

    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = -1.0
    robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
    action = action_primitives._empty_action(follow_arm_targets=False)
    print("action: ", action)
    env.step(action)
    for i in range(40):
        print(i, robot._controllers["gripper_right"]._goal)
        og.sim.step()
    print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])

    
    for i in range(3000):
        og.sim.step()

    # for i in range(300):
    #     env.step(action)
    #     print(i, robot._controllers["arm_right"]._goal)
    #     og.sim.step()

    # if pos_error > 0.2 or np.rad2deg(orn_error) > 20:
    og.sim.load_state(state)

og.shutdown()