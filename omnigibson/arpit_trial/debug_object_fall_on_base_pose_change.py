import os
import yaml
import  pdb

import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R

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
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    }
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# Setting Tiago gripper's friction high so that objects don't slip
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

# og.clear()
og.sim.restore(["moma_pick_and_place/episode_00000_before_place.json"])

box = env.scene.object_registry("name", "box")
box.root_link.mass = 1e-2
print("box.mass: ", box.mass)

for _ in range(100):
    og.sim.step()

base_pose = robot.get_position_orientation()
base_pos = base_pose[0]
base_noise = th.tensor([-0.1, 0.0, 0.0])
base_pos += base_noise 

base_yaw = R.from_quat(base_pose[1]).as_euler('XYZ', degrees=True)[2]
base_yaw_noise = 45
base_yaw += base_yaw_noise
r_euler = R.from_euler('z', base_yaw, degrees=True) 
r_quat = R.as_quat(r_euler)

action = th.zeros(robot.action_dim)
print("action: ", action)
base_x_vel = np.random.uniform(-0.1, 0.05)
base_y_vel = np.random.uniform(-0.1, 0.1)
base_yaw_vel = np.random.uniform(-0.2, 0.2)
# action[:3] = th.tensor([0.0, 0.0, 0.2])
env.step(action)
for _ in range(30):
    og.sim.step()

action = th.zeros(robot.action_dim)
env.step(action)


# robot.set_position_orientation(base_pos, r_quat)

for _ in range(500):
    og.sim.step()

og.shutdown()