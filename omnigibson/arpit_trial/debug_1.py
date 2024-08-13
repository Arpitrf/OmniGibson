import os
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from argparse import ArgumentParser
import time
from scipy.spatial.transform import Rotation as R

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def convert_delta_to_absolute(action, robot):
    current_pose = action_primitives._get_pose_in_robot_frame((robot.get_eef_position(), robot.get_eef_orientation()))
    current_pos = current_pose[0]
    current_orn = current_pose[1]
    # target_pos = 


def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma")
    # parser.add_argument('--f_name', type=str)
    # parser.add_argument('--start_frame', type=int, default=1)
    # parser.add_argument('--hand', type=str, default='left')
    parser.add_argument('--save_traj', action='store_true', default=False)
    return parser

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def print_dict_keys(dictionary, indent=0):
    for key, value in dictionary.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_dict_keys(value, indent + 4)


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    
    It loads Rs_int with a Fetch robot, and the robot picks and places a bottle of cologne.
    """
    # Initializations
    np.random.seed(1)
    args = config_parser().parse_args()

    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "size": 0.05,
            "position": [-0.3, -0.8, 0.5]
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1]
        }
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]    

    # --------- Comment/Uncomment this ----------
    # og.sim.restore('sim_state_block_pick.json')
    
    state = og.sim.dump_state()
    og.sim.stop()
    
    # trying increasing friction in fingers
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=400.0,
        dynamic_friction=400.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        print("arm, links: ", arm, links)
        for link in links:
            # print("link: ", link)
            for msh in link.collision_meshes.values():
                print("msh: ", msh)
                msh.apply_physics_material(gripper_mat)

    for obj in scene.objects:
        if obj.name == 'box':
            obj_box = obj
            from omni.isaac.core.materials import PhysicsMaterial
            primitive_mat = PhysicsMaterial(
                prim_path=f"{obj_box.prim_path}/box_mat",
                name="box_material",
                static_friction=400.0,
                dynamic_friction=400.0,
                restitution=None,
            )
            print("obj: ", type(obj_box))
            for link in obj_box.links.values():
                # print("link: ", link)
                for msh in link.collision_meshes.values():
                    print("msh: ", msh)
                    msh.apply_physics_material(primitive_mat)
    
    og.sim.play()
    og.sim.load_state(state)
    # env.reset()
    # robot.reset()

    # # added by Arpit: set current_pose of eef
    # pos, orn = robot.get_relative_eef_pose()
    # robot.controllers['arm_left'].set_current_stable_pose(pos, orn)

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

    # checking
    print("box static friction: ", obj_box.links['base_link'].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())
    print("box dynamic friction: ", obj_box.links['base_link'].collision_meshes['collisions'].get_applied_physics_material().get_dynamic_friction())
    print("robot left gripper left finger: ", robot.finger_links['left'][0].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())
    print("robot left gripper right finger: ", robot.finger_links['left'][1].collision_meshes['collisions'].get_applied_physics_material().get_static_friction())

    max_steps = -1
    step = 0
    while step != max_steps:
        flag = False
        action, keypress_str = action_generator.get_teleop_action()

        if keypress_str == 'TAB':
            flag = True
            # break
        obs, _, _, _ = env.step(action=action)
        step += 1


    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()