import os
import yaml
import  pdb

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

def custom_reset(env, robot): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset(head_joints_pos=head_joints)

def execute_controller(ctrl_gen, env, robot, gripper_closed):
    for action in ctrl_gen:
        if gripper_closed:
            # if left hand is ik
            # action[18] = -1
            # if left hand is joint controller
            action[20] = -1
        else: 
            # action[18] = 1
            action[20] = 1
        env.step(action)

        # debugging:
        current_pos_world = robot.eef_links["right"].get_position_orientation()
        print("current_pose_world: ", current_pos_world[0])


def primitive():
    gripper_closed = False
    # move hand to a pose
    target_pose = (th.tensor([0.2129, 0.4976, 0.3351]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True, pos_thresh=0.01), env, robot, gripper_closed) 
   
    for _ in range(40):
        og.sim.step()
    current_pos_world = robot.eef_links["right"].get_position_orientation()
    print("move hand down completed. Final right eef pose reached: ", current_pos_world)

    # 3. Perform grasp
    gripper_closed = True
    action = action_primitives._empty_action()
    # if left hand is IK
    # action[18] = -1
    # if left has is joint controller
    action[20] = -1
    execute_controller([action], env, robot, gripper_closed)
    # step the simulator a few steps to let the gripper close completely
    for _ in range(40):
        og.sim.step()
        
    # move hand up
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.5])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True), env, robot, gripper_closed)

    # input("Press enter to continue =============================")

    # # move base
    # # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    # target_base_pose = th.tensor([0.4256, 0.0257, 0.0])
    # execute_controller(action_primitives._navigate_to_pose_direct(target_base_pose), env, robot, gripper_closed)

    # input("Press enter to continue =============================")

    # # move hand to a pose
    # target_pose =  (th.tensor([ 1.1888, -0.1484,  0.8187]), th.tensor([-0.0489, -0.0063,  0.5555,  0.8301]))
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), env, robot, gripper_closed)





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
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
        # "orientation": [0.0004835024010390043,
        #             -0.00029672126402147114,
        #             -0.11094563454389572,
        #             0.9938263297080994]
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

custom_reset(env, robot)

# scene_initial_state = env.scene._initial_state
# base_yaw = 90
# r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
# r_quat = R.as_quat(r_euler)
# scene_initial_state['object_registry']['robot0']['root_link']['ori'] = r_quat

# robot.set_orientation(r_quat)

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
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# # Teleop
# max_steps = -1 
# step = 0
# # # pdb.set_trace()
# while step != max_steps:
#     action, keypress_str = action_generator.get_teleop_action()
#     env.step(action=action)
#     if keypress_str == 'TAB':
#         right_eef_pose = robot.get_relative_eef_pose(arm='right')
#         right_eef_pose_world = robot.eef_links["right"].get_position_orientation()
#         base_pose = robot.get_position_orientation()
#         print("right_eef_pose: ", right_eef_pose)
#         print("right_eef_pose_world: ", right_eef_pose_world)
#         print("base_pose: ", base_pose)
#     step += 1

primitive()

for _ in range(5000):
    og.sim.step()

# Always shut down the environment cleanly at the end
og.clear()



