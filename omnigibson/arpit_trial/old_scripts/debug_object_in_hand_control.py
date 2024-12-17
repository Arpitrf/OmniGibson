import os
import yaml
import  pdb
import argparse

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

def custom_reset(env, robot): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset()

def execute_controller(ctrl_gen, env, robot, gripper_closed):
    for action in ctrl_gen:
        if action == 'Done':
            continue
        if gripper_closed:
            # if left hand is ik
            # action[18] = -1
            # if left hand is joint controller
            action[20] = -1
        else: 
            # action[18] = 1
            action[20] = 1
        # print("action: ", action)
        env.step(action)


parser = argparse.ArgumentParser()
parser.add_argument("--delta_pos", action="store_true")
parser.add_argument("--no_box_object", action="store_true", default=False)
parser.add_argument('--default_arm_pose', type=str, default='horizontal')
args = parser.parse_args()

config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

if args.default_arm_pose == 'horizontal':
    config['robots'][0]['default_arm_pose'] = 'horizontal'
else:
    config['robots'][0]['default_arm_pose'] = 'vertical'


if args.delta_pos:
    config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_delta_ori'
else:
    config['robots'][0]['controller_config']['arm_right']['mode'] = 'absolute_pose'

config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

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
]

if not args.no_box_object:
    box =  {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    }
    config["objects"].append(box)


env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

custom_reset(env, robot)

# Setting friction
state = og.sim.dump_state()
og.sim.stop()
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


# box = env.scene.object_registry("name", "box")
# box.root_link.mass = 1e-2
# print("obj.mass: ", obj.mass)

normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx[arm]]
print("normalized_qpos: ", normalized_qpos)

# ============ Move hand to grasp pose =============
curr_eef_pose = robot.get_relative_eef_pose(arm='right')
print("intial_pose: ", curr_eef_pose[0])
if args.default_arm_pose == 'horizontal':
    target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), curr_eef_pose[1])
else:
    target_pose = (th.tensor([ 0.4976, -0.0829,  0.5246]), curr_eef_pose[1])

action = th.zeros(robot.action_dim)
if args.delta_pos:
    # pose_delta_ori mode
    delta_pos = target_pose[0] - curr_eef_pose[0] 
    print("delta_pos: ", delta_pos)
    delta_orn = th.zeros(3)
    action[14:17] = th.tensor(delta_pos)
    action[17:20] = th.tensor(delta_orn)
else:
    # absolute_pose mode
    orn = R.from_quat(target_pose[1]).as_rotvec()
    action[14:17] = th.tensor(target_pose[0])
    action[17:20] = th.tensor(orn)

env.step(action)
for _ in range(100):
    og.sim.step()

reached_right_eef_pose = robot.get_relative_eef_pose(arm='right')
print("desired_right_eef_pose, reached_right_eef_pose: ", target_pose[0], reached_right_eef_pose[0])
# ===========================================

# ============ Close gripper =============
action = th.zeros(robot.action_dim)
# only if absolute pose
if not args.delta_pos:
    orn = R.from_quat(target_pose[1]).as_rotvec()
    action[14:17] = th.tensor(target_pose[0])
    action[17:20] = th.tensor(orn)
action[20] = -1

env.step(action)
for _ in range(100):
    og.sim.step()
# ====================================== 
    
# ============ Move hand to post-grasp pose =============
curr_eef_pose = robot.get_relative_eef_pose(arm='right')
target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.4]), curr_eef_pose[1])
action = th.zeros(robot.action_dim)
if args.delta_pos:
    # pose_delta_ori mode
    delta_pos = target_pose[0] - curr_eef_pose[0] 
    print("delta_pos: ", delta_pos)
    delta_orn = th.zeros(3)
    action[14:17] = th.tensor(delta_pos)
    action[17:20] = th.tensor(delta_orn)
    action[20] = -1
else:
    # absolute_pose mode
    orn = R.from_quat(target_pose[1]).as_rotvec()
    action[14:17] = th.tensor(target_pose[0])
    action[17:20] = th.tensor(orn)
    action[20] = -1

env.step(action)
for _ in range(300):
    og.sim.step()

reached_right_eef_pose = robot.get_relative_eef_pose(arm='right')
print("desired_right_eef_pose, reached_right_eef_pose: ", target_pose[0], reached_right_eef_pose[0])
pos_error = np.linalg.norm(target_pose[0] - reached_right_eef_pose[0])
orn_error = T.get_orientation_diff_in_radian(target_pose[1], reached_right_eef_pose[1])
print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
# =========================================================

for _ in range(500):
    og.sim.step()

# Always shut down the environment cleanly at the end
og.shutdown()