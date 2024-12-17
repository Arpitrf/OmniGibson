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
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.contact_bodies import ContactBodies
from memory import Memory
from motion_utils import MotionUtils

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=1)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

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
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    }
]

env = og.Environment(configs=config)

ep = "00547"
og.sim.restore([f"place_in_shelf_data_high_noise/episode_{ep}_start.json"])


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

# getting all objects in scene
# env.scene.objects
shelf = env.scene.object_registry("name", "shelf")
shelf.root_link.mass = 1e3
box = env.scene.object_registry("name", "box")
box.root_link.mass = 1e-2
print("box.mass: ", box.mass)

f = h5py.File("/home/arpit/test_projects/OmniGibson/place_in_shelf_data_high_noise/dataset.hdf5", "r")
actions = np.array(f[f"data/episode_{ep}/actions/actions"])
print("actions: ", actions)

for _ in range(50):
    og.sim.step()

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
motion_utils = MotionUtils(env, robot, action_primitives)

for action in actions:
    motion_utils.move_primitive(action)

# # =============================== Teleop ===============================
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

# max_steps = -1 
# step = 0
# box = env.scene.object_registry("name", "box")
# while step != max_steps:
#     action, keypress_str = action_generator.get_teleop_action()
    
#     action[robot.gripper_action_idx["right"]] = -1
#     env.step(action=action)
#     arm_in_collision = detect_robot_collision_in_sim(robot, filter_objs=[box])
#     for i in range(1,8):
#         lis = robot.links[f"arm_right_{i}_link"].contact_list()
#         if len(lis) > 0:
#             print(f"arm_right_{i}_link in contact at step {step}: ", lis)
#     # print(f"arm_in_collision at step {step}: ", arm_in_collision)
#     step += 1
#     if keypress_str == 'TAB':
#         breakpoint()
# # ========================================================================
            

og.shutdown()


