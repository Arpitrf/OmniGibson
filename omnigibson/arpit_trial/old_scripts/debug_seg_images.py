import os
import yaml

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T



def randomize_robot():
    # move base
    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = -1
    print("action: ", action)
    # base_x_vel = np.random.uniform(-0.01, 0.05)
    base_x_vel = np.random.uniform(0.08, 0.13)
    base_y_vel = np.random.uniform(-0.1, 0.1)
    base_yaw_vel = np.random.uniform(-0.01, 0.01) # 0.2 originally
    # action[:3] = th.tensor([0.0, 0.0, 0.2])
    action[:3] = th.tensor([base_x_vel, base_y_vel, base_yaw_vel])
    env.step(action)
    timesteps = np.random.randint(15, 40)
    for _ in range(timesteps): # was 30 before
        og.sim.step()
    
    # Randomizing head pose
    # default_head_joints = np.array([-0.20317451, -0.7972661])
    # default_head_joints = np.array([-0.5031718015670776, -0.9972541332244873])
    default_head_joints = np.array([-0.503, -0.797])
    noise_1 = np.random.uniform(-0.05, 0.05, 1)
    noise_2 = np.random.uniform(-0.05, 0.1, 1)
    noise = np.concatenate((noise_1, noise_2))
    head_joints_pos = default_head_joints + noise
    # head_joints_pos = np.array([0.0, -0.5])
    head_joints_pos = th.from_numpy(head_joints_pos)
    head_joints_pos = th.tensor(head_joints_pos, dtype=th.float32)
    robot.set_joint_positions(head_joints_pos, indices=robot.camera_control_idx)

    action = th.zeros(robot.action_dim)
    action[robot.gripper_action_idx["right"]] = -1
    env.step(action)

    for _ in range(50):
        og.sim.step()

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

# og.clear()
# og.sim.restore(["episode_00000_before_place.json"])
# og.sim.restore(["moma_pick_and_place/episode_00000_start.json"])
# og.sim.restore(["place_start.json"])

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

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

for _ in range(100):
    og.sim.step()

state = og.sim.dump_state(serialized=False)
    
# Randomize base pose and head pose a bit
randomize_robot()

robot_name = env.robots[0].name
obs, obs_info = env.get_obs()
print("seg_semantic: ", obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_semantic'])
seg_semantic = obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_semantic'].cpu().numpy()
plt.imshow(seg_semantic)
plt.show()


og.shutdown()


