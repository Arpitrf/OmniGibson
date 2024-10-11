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

from scipy.spatial.transform import Rotation as R
from memory import Memory
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson import object_states

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

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

set_all_seeds(seed=1)
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

# og.sim.restore(["moma_pick_and_place/episode_00001_start.json"])
og.sim.restore(["moma_pick_and_place/temp.json"])

correct_gripper_friction()

for _ in range(50):
    og.sim.step()


shelf = env.scene.object_registry("name", "shelf")
shelf_pos_orn = shelf.get_position_orientation()

state = og.sim.dump_state()

for _ in range(10000):
    og.sim.step()

for i in range(20):

    orn_noise = np.random.uniform(-1.0, 1.0, 4)
    pos_noise = np.random.uniform(0.5, 1.0, 2)

    # orn = shelf_pos_orn[1] + th.tensor(orn_noise).type(th.FloatTensor)
    # orn = orn / th.linalg.norm(orn)
    # shelf.set_position_orientation(shelf_pos_orn[0] + th.tensor([pos_noise[0], pos_noise[1], 0.0]).type(th.FloatTensor), orn)

    # print("11")
    # for _ in range(50):
    #     og.sim.step()

    # state = og.sim.dump_state()

    print("22")
    # # Teleop
    max_steps = -1 
    step = 0
    while step != max_steps:
        action, keypress_str = action_generator.get_teleop_action()
        env.step(action=action)
        if keypress_str == 'TAB':
            break
        step += 1
        
    # # test
    # robot.set_position_orientation(position=th.tensor([-2.0, 0.0, 0.0]))
    # for _ in range(100):
    #     og.sim.step()
    
    print("Resetting to original state")
    shelf.wake()
    og.sim.load_state(state)
    shelf.wake()
    og.sim.step()
    og.sim.step()

    # print("33")
    # for _ in range(50):
    #     og.sim.step()

    # print("Resetting the shelf to original state by using set_position_orientation")
    # shelf.set_position_orientation(shelf_pos_orn[0], shelf_pos_orn[1])

    for _ in range(100):
        og.sim.step()