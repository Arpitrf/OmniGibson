import os
import yaml
import argparse

import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R

DELTA_POS = True

def custom_reset(env, robot):
    scene_initial_state = env.scene._initial_state

    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True)  # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset()


config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"
config["robots"][0]["name"] = "robot0"
config["robots"][0]["controller_config"]["arm_right"] = {
    "name": "InverseKinematicsController",
    "command_input_limits": None,
    "command_output_limits": None,
}
config["robots"][0]["controller_config"]["gripper_right"] = {
    "name": "MultiFingerGripperController",
}

if DELTA_POS:
    config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_delta_ori'
else:
    config['robots'][0]['controller_config']['arm_right']['mode'] = 'absolute_pose'
    config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
    config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None
# config['robots'][0]['controller_config']['arm_left']['use_delta_commands'] = True


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
        "position": [0, 0.6, 0.3],
        "orientation": [0, 0, 0, 1]
    },
    {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

custom_reset(env, robot)

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

for _ in range(25):
    og.sim.step()

obj = env.scene.object_registry("name", "box")
# obj_pos = obj.get_position_orientation()[0]

# Convert obj pose into robot frame
obj_rel_pos, _ = T.relative_pose_transform(*obj.get_position_orientation(), *robot.get_position_orientation())

# obj.root_link.mass = 1e-2
# print("obj.mass: ", obj.mass)

# ============ Move hand to grasp pose =============
init_eef_pose = robot.get_relative_eef_pose(arm='right')
print("intial_pose: ", init_eef_pose[0])
# target_pose = (th.tensor([0.4976, -0.2129, 0.4346]), curr_eef_pose[1])
target_pose = (obj_rel_pos + th.tensor([0, 0, 0.05]), init_eef_pose[1])
target_pose_above = (obj_rel_pos + th.tensor([0, 0, 0.25]), init_eef_pose[1])

action = th.zeros(robot.action_dim)
# Keep right gripper open
action[20] = 1.0

for pose in (target_pose_above, target_pose):

    curr_eef_pose = robot.get_relative_eef_pose(arm='right')

    if DELTA_POS:
        # pose_delta_ori mode
        delta_pos = pose[0] - curr_eef_pose[0]
        print("delta_pos: ", delta_pos)
        delta_orn = th.zeros(3)
        action[14:17] = th.tensor(delta_pos)
        action[17:20] = th.tensor(delta_orn)
    else:
        # absolute_pose mode
        orn = R.from_quat(pose[1]).as_rotvec()
        action[14:17] = th.tensor(target_pose[0])
        action[17:20] = th.tensor(orn)

    env.step(action)
    for _ in range(50):
        og.sim.step()

reached_right_eef_pose = robot.get_relative_eef_pose(arm='right')
print("desired_right_eef_pose, reached_right_eef_pose: ", target_pose[0], reached_right_eef_pose[0])
# ===========================================

# ============ Close gripper =============
action = th.zeros(robot.action_dim)
# only if absolute pose
if not DELTA_POS:
    orn = R.from_quat(target_pose[1]).as_rotvec()
    action[14:17] = th.tensor(target_pose[0])
    action[17:20] = th.tensor(orn)
action[20] = -1

env.step(action)
for _ in range(20):
    og.sim.step()
# ======================================

# ============ Move hand to post-grasp pose =============
curr_eef_pose = robot.get_relative_eef_pose(arm='right')
action = th.zeros(robot.action_dim)
if DELTA_POS:
    # pose_delta_ori mode
    delta_pos = target_pose_above[0] - curr_eef_pose[0]
    print("delta_pos: ", delta_pos)
    delta_orn = th.zeros(3)
    action[14:17] = th.tensor(delta_pos)
    action[17:20] = th.tensor(delta_orn)
else:
    # absolute_pose mode
    orn = R.from_quat(target_pose[1]).as_rotvec()
    action[14:17] = th.tensor(target_pose[0])
    action[17:20] = th.tensor(orn)

action[20] = -1
env.step(action)
for _ in range(50):
    og.sim.step()

reached_right_eef_pose = robot.get_relative_eef_pose(arm='right')
print("desired_right_eef_pose, reached_right_eef_pose: ", target_pose_above[0], reached_right_eef_pose[0])
# =========================================================

# for _ in range(500):
#     og.sim.step()

# Always shut down the environment cleanly at the end
# og.shutdown()


