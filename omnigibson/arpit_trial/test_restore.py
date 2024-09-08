import os
import yaml
import pickle

import numpy as np
import omnigibson as og


def main():
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to run a grocery shopping task
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "Cube"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",
            "position": [-0.3, -1.1, 0.5],
            "orientation": [0, 0, 0, 1],
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    for _ in range(100):
        og.sim.step()

    print("Reloading state")
    # og.clear()
    og.sim.restore("/home/arpit/test_projects/OmniGibson/prior/episode_00012_end.json")

    for _ in range(1000):
        og.sim.step()

if __name__ == "__main__":
    main()