import os
import yaml
import  pdb

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

def execute_controller(ctrl_gen, env, robot):
    idx = list(robot.joints.keys()).index("arm_right_7_joint")
    for action in ctrl_gen:
        proprio = robot._get_proprioception_dict()
        print("joint_forces: ", robot.get_joint_forces().shape)
        joint_efforts = robot.get_joint_efforts()
        # print("applied effor and measured effort at joint 7: ", step, proprio['joint_qeffort'][idx], robot.get_joint_efforts()[idx])
        env.step(action)

# decrypt_file('/home/arpit/test_projects/OmniGibson/data/datasets/og_dataset/objects/fridge/dszchb/usd/dszchb.encrypted.usd', '/home/arpit/test_projects/OmniGibson/fridge_dszchb.usd')

config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
obj_cfg = dict(
    type="DatasetObject",
    name="bottom_cabinet",
    category="bottom_cabinet",
    model="bamfsz",
    position=[0.9, 0, 1.0],
    scale=[2.0, 1.0, 2.0],
    orientation=rot_quat,
    )
config["objects"] = [obj_cfg]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

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

# Loop control until user quits
max_steps = -1 
step = 0

# collision checks: detect_robot_collision_in_sim, detect_robot_collision()
# TODO: collision check separately for the arm and rest of the body
# FT value

idx = list(robot.joints.keys()).index("arm_right_7_joint")
# pdb.set_trace()
while step != max_steps:
    action = action_generator.get_teleop_action()
    env.step(action=action)
    step += 1
    proprio = robot._get_proprioception_dict()
    # print(proprio['joint_qeffort'].shape)
    # print(robot.joints.keys())
    # if step % 100 == 0:
    robot.get_joint_forces()
    if robot.get_joint_efforts()[idx].item() > 0.001:
        # print("effort at joint 7: ", proprio['joint_qeffort'][idx])
        print("applied effor and measured effort at joint 7: ", step, proprio['joint_qeffort'][idx], robot.get_joint_efforts()[idx])


# # move hand to a pose
# action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
# # move hand 40 cm front
# curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
# new_pos = curr_pos + th.tensor([0.4, 0.0, 0.0])
# target_pose = (new_pos, curr_orn)
# execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True), env, robot)

for _ in range(5000):
    og.sim.step()

# Always shut down the environment cleanly at the end
og.clear()

# # Place the object so it rests on the floor
# obj = env.scene.object_registry("name", "obj")
# center_offset = obj.get_position() - obj.aabb_center + np.array([0, 0, obj.aabb_extent[2] / 2.0])
# obj.set_position(center_offset)

# for _ in range(50):
#     og.sim.step()
# obs, obs_info = env.get_obs()
# seg_semantic = obs['robot0']['robot0:eyes:Camera:0']['seg_semantic'].cpu()
# seg_instance = obs['robot0']['robot0:eyes:Camera:0']['seg_instance'].cpu()
# seg_instance_id = obs['robot0']['robot0:eyes:Camera:0']['seg_instance_id'].cpu()
# print("seg_instance_id.shape: ", seg_instance_id.shape)
# fig, ax = plt.subplots(1,3)
# ax[0].imshow(seg_semantic)
# ax[1].imshow(seg_instance)
# ax[2].imshow(seg_instance_id)
# plt.show()

# for _ in range(5000):
#     og.sim.step()


