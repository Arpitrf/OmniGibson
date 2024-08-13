import os
import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.
    
    It loads Rs_int with a Fetch robot, and the robot picks and places a bottle of cologne.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    ## Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    # config["scene"]['scene_file'] = 'sim_state_block_push.json'
    config["objects"] = [
        {
            "type": "PrimitiveObject",
            "name": "box",
            "primitive_type": "Cube",
            "rgba": [1.0, 0, 0, 1.0],
            "scale": [0.1, 0.05, 0.1],
            # "size": 0.05,
            "position": [-0.5, -0.7, 0.5],
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

    # Obtian object pose
    # obj_info = scene.get_objects_info()
    # print("obj_info: ", obj_info)
    scene_state = scene.dump_state(serialized=False)
    print("box: ", scene_state['object_registry']['box']['root_link']) 

    # # Allow user to move camera more easily
    # og.sim.enable_viewer_camera_teleoperation()

    # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=True, always_track_eef=True)

    # # Grasp of cologne
    # grasp_obj = scene.object_registry("name", "cologne")
    # print("Executing controller")
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    # print("Finished executing grasp")

    # # Place cologne on another table
    # print("Executing controller")
    # table = scene.object_registry("name", "table")
    # execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), env)
    # print("Finished executing place")

if __name__ == "__main__":
    main()