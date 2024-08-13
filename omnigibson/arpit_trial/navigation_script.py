import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


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
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.

    It loads Rs_int with a Fetch robot, and the robot picks and places an apple.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to run a grocery shopping task
    config["scene"]["scene_model"] = "Rs_int"

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "conference_table",
            "model": "qzmjrj", 
            "position": [(-0.5 + 1) - 0.5, (-0.7 - 1.5) + 0.5, 0.4],
            "scale": [1, 1, 0.6],
            "orientation": R.from_euler("xyz", [0, 0, -3.14/2]).as_quat()
        },
        {
            "type": "DatasetObject",
            "name": "apple", 
            "category": "apple",
            "model": "omzprq", 
            "position": [-0.1, -0.9, 0.6]
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    cabinet = scene.object_registry("name", "bottom_cabinet_slgzfc_0")
    apple = scene.object_registry("name", "apple")

    # Navigate to apple
    print("Executing controller")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, apple), env)
    print("Finished executing grasp")

if __name__ == "__main__":
    main()