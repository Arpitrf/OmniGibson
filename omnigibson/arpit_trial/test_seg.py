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
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.objects import PrimitiveObject
from omnigibson.utils.python_utils import nums2array


config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# config["robots"][0]["controller_config"]["arm_right"]["name"] = "OperationalSpaceController"

# config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
# config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'absolute_pose'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, -30.0]
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
    # {
    #     "type": "PrimitiveObject",
    #     "name": "box",
    #     "primitive_type": "Cube",
    #     "rgba": [1.0, 0, 0, 1.0],
    #     "scale": [0.1, 0.05, 0.1],
    #     # "visual_only": True,
    #     # "size": 0.05,
    #     "mass": 1e-6,
    #     "position": [0.1, 0.5, 0.5],
    #     "orientation": box_quat
    # },
    {
        "type": "DatasetObject",
        "name": "can_of_baking_mix",
        "category": "can_of_baking_mix",
        "model": "blrqqz", 
        # "scale": [0.6, 0.6, 0.8],
        "position": [0.1, 0.5, 0.5],
        "orientation": [0, 0, 0, 1]
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
print(robot.name)

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=200.0,
    dynamic_friction=200.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)

og.sim.play()
og.sim.load_state(state)

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# pdb.set_trace()
# obj = env.scene.object_registry("name", "box")
obj = env.scene.object_registry("name", "can_of_baking_mix")
obj.root_link.mass = 1e-1
shelf = env.scene.object_registry("name", "shelf")
# shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))
shelf.root_link.mass = 1e3
# print("obj.mass: ", obj.mass)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

eef_marker = PrimitiveObject(
            relative_prim_path=f"/marker",
            name="marker",
            primitive_type="Sphere",
            radius=0.03,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0],
        )
env.scene.add_object(eef_marker)

for _ in range(20):
    og.sim.step()


# =============================== Teleop ===============================
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
robot_name = env.robots[0].name
while step != max_steps:
    action, keypress_str = action_generator.get_teleop_action()    
    obs, reward, terminated, truncated, info = env.step(action=action)
    if keypress_str == 'TAB':
        fig, ax = plt.subplots(2,2)
        seg_semantic = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_semantic'].cpu().numpy()
        seg_instance = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance'].cpu().numpy()
        seg_instance_id = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'].cpu().numpy()
        rgb = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['rgb'].cpu().numpy()
        ax[0, 0].imshow(rgb[:, :, :3])
        ax[0, 1].imshow(seg_semantic)
        ax[1, 0].imshow(seg_instance)
        ax[1, 1].imshow(seg_instance_id)
        plt.show()
        breakpoint()
    step += 1
# ========================================================================

for _ in range(500):
    og.sim.step()

# Always shut down the environment cleanly at the end
# og.clear()