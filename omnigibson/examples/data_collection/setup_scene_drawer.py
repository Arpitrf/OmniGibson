import os
import pdb
import yaml

import torch as th
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy

from scipy.spatial.transform import Rotation as R
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

gm.USE_GPU_DYNAMICS = True

def main():
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["load_object_categories"] = ["bottom_cabinet", "floors"]
    cfg["task"]["activity_name"] = "test_open_drawer"
    cfg["task"]["online_object_sampling"] = True
    cfg["env"]["flatten_obs_space"] = True
    cfg["env"]["action_frequency"] = 30
    cfg["env"]["rendering_frequency"] = 30
    cfg["env"]["physics_frequency"] = 120
    cfg["robots"][0]["default_reset_mode"] = "untuck"
    
    # rot_euler = [0.0, 0.0, -90.0]
    # rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
    # obj_cfg = dict(
    #     type="DatasetObject",
    #     name="bottom_cabinet",
    #     category="bottom_cabinet",
    #     model="bamfsz",
    #     position=[1.5, 0, 0.7],
    #     scale=[2.0, 1.0, 2.0],
    #     orientation=rot_quat,
    # )
    # cfg["objects"] = [obj_cfg]
    
    # Load the environment
    env = og.Environment(configs=cfg)

    for _ in range(10):
        og.sim.step()

    # pdb.set_trace()

    # Manually move the robot and the objects to the desired initial poses by calling obj.set_position_orientation()
    robot = env.robots[0]
    pos, orn = robot.get_position_orientation()
    target_pos = pos + th.tensor([0.2, 0.0, 0.0])
    robot.set_position_orientation(target_pos, orn)

    for _ in range(10):
        og.sim.step()

    # Save the scene cache
    env.task.save_task(path="Rs_int_task_test_open_drawer_0_0_template3.json")


if __name__ == "__main__":
    main()
