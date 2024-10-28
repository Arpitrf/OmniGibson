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


def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    obs, info = env.get_obs()
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
    return obs, info


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
    
    # base_yaw = 90
    # r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    # r_quat = R.as_quat(r_euler)
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # # randomizing base pos
    # base_pos = np.array([-0.05, -0.4, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

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


set_all_seeds(seed=1)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
# config["scene"] = dict()
# config["scene"]["type"] = "Scene"
# config["scene"]["scene_model"] = "Rs_int"
config["scene"]["load_object_categories"] = ["floors", "ceilings", "coffee_table", "breakfast_table", "pot_plant", "laptop", "floor_lamp", "table_lamp"]

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_absolute_ori'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, 0.0]
box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
config["objects"] = [
    # {
    #     "type": "DatasetObject",
    #     "name": "shelf",
    #     "category": "shelf",
    #     "model": "eniafz",
    #     "position": [1.5, 0, 1.0],
    #     "scale": [2.0, 2.0, 1.0],
    #     "orientation": rot_quat,
    # },   
    # {
    #     "type": "DatasetObject",
    #     "name": "coffee_table",
    #     "category": "coffee_table",
    #     "model": "fqluyq",
    #     # "scale": [0.3, 0.3, 0.3],
    #     "position": [0, 0.6, 0.3],
    #     "orientation": [0, 0, 0, 1]
    # },
    # {
    #     "type": "PrimitiveObject",
    #     "name": "box",
    #     "primitive_type": "Cube",
    #     "rgba": [1.0, 0, 0, 1.0],
    #     "scale": [0.1, 0.05, 0.1],
    #     "position": [0.1, 0.5, 0.5],
    #     "orientation": box_quat
    # },
    {
        "type": "DatasetObject",
        "name": "box_of_baking_powder",
        "category": "box_of_baking_powder",
        "model": "vzgrlv",
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# og.sim.restore(["moma_pick_and_place/temp.json"])
# og.sim.restore(["temp2.json"])


coffee_table = env.scene.object_registry("name", "coffee_table_fqluyq_0")
breakfast_table = env.scene.object_registry("name", "breakfast_table_skczfi_0")

# fridge = env.scene.object_registry("name", "fridge_xyejdx_0")
laptop = env.scene.object_registry("name", "laptop_nvulcs_0")
table_lamp = env.scene.object_registry("name", "table_lamp_xbfgjc_0")
box = env.scene.object_registry("name", "box_of_baking_powder")
pot_plant = env.scene.object_registry("name", "pot_plant_jatssq_0")
pot_plant2 = env.scene.object_registry("name", "pot_plant_jatssq_1")
floor_lamp = env.scene.object_registry("name", "floor_lamp_vdxlda_0")

box.states[object_states.OnTop].set_value(breakfast_table, True)

# # Set viewer camera
# og.sim.viewer_camera.set_position_orientation(
#     th.tensor([-0.7563,  1.1324,  1.0464]),
#     th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
# )

scene = env.scene
robot = env.robots[0]
correct_gripper_friction()
# shelf = env.scene.object_registry("name", "shelf")
# shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

init_pose = robot.get_relative_eef_pose(arm='right')


for _ in range(300):
    # for name, ctrl in robot._controllers.items():
    #     print(name, robot._controllers[name]._goal)
    # print(robot._controllers["arm_right"]._goal)
    og.sim.step()

post_eef_pose = robot.get_relative_eef_pose(arm='right')
pos_error = np.linalg.norm(post_eef_pose[0] - init_pose[0])
orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], init_pose[1])
print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

grasp_action = -1
init_pose = robot.get_relative_eef_pose(arm='right')

# # If want to run without the object
# robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
# for _ in range(20):
#     og.sim.step()

state = og.sim.dump_state()

objs = [pot_plant2, laptop, box, pot_plant, floor_lamp]

for obj in objs:
    print(f"Navigating to {obj.name}")
    input()
    execute_controller(action_primitives._navigate_to_obj(obj),
                        env, robot, grasp_action=1.0)

for i in range(50000):
    og.sim.step()


og.shutdown()
