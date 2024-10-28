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
            print("link: ", link)
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
og.sim.restore(["temp2.json"])

scene = env.scene
robot = env.robots[0]

correct_gripper_friction()
shelf = env.scene.object_registry("name", "shelf")
shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

for _ in range(50):
    og.sim.step()

state = og.sim.dump_state()
for _ in range(10):
    print("wait start")
    for _ in range(100):
        og.sim.step()
    print("wait end")
    # robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
    # action = action_primitives._empty_action()
    action = th.zeros(robot.n_dof)
    action[robot.gripper_action_idx["right"]] = 1.0
    env.step(action)
    for _ in range(40):
        og.sim.step()
    print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])
    og.sim.load_state(state)


og.shutdown()