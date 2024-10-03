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
    curr_eef_pose = robot.get_relative_eef_pose(arm='right')
    # move hand to a pose 
    # target_pose = (th.tensor([0.2129, 0.4976, 0.3351]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
    target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), curr_eef_pose[1])
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False, pos_thresh=0.01), env, robot, gripper_closed) 
   
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
    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.3])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, in_world_frame=False, ignore_failure=True), env, robot, gripper_closed)

    reached_right_eef_pose = robot.get_relative_eef_pose(arm='right')
    print("desired_right_eef_pose, reached_right_eef_pose: ", target_pose[0], reached_right_eef_pose[0])
    pos_error = np.linalg.norm(target_pose[0] - reached_right_eef_pose[0])
    orn_error = T.get_orientation_diff_in_radian(target_pose[1], reached_right_eef_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_absolute_ori'
config['robots'][0]['default_arm_pose'] = 'diagonal45'

# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None


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
    # {
    #     "type": "PrimitiveObject",
    #     "name": "box",
    #     "primitive_type": "Cube",
    #     "rgba": [1.0, 0, 0, 1.0],
    #     "scale": [0.1, 0.05, 0.1],
    #     # "size": 0.05,
    #     "position": [0.14, 0.53, 0.5],
    #     "orientation": box_quat
    #     # "orientation": [0.0004835024010390043,
    #     #             -0.00029672126402147114,
    #     #             -0.11094563454389572,
    #     #             0.9938263297080994]
    # },
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

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

primitive()

for _ in range(500):
    og.sim.step()

# Always shut down the environment cleanly at the end
og.clear()



