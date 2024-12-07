import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm

config = dict()
scene_cfg = dict()
scene_cfg["type"] = "Scene"

robot0_cfg = {
    "type": "Tiago",
    "name": "tiago",
    # "obs_modalities": "rgb",
    "self_collisions": False,
    "action_normalize": False,
    "action_type": "continuous",
    "grasping_mode": "physical",
    "rigid_trunk": False,
    "default_trunk_offset": 0.15,
    "controller_config": {
        "base": {
            "name": "JointController",
        },
        "arm_left": {
            "name": "JointController",
        },
        "arm_right": {
            "name": "JointController",
        },
        # "gripper_left": {
        #     "name": "JointController",
        # },
        "gripper_right": {
            "name": "JointController",
        },
    },
}

# Compile config
cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

# Create the environment
env = og.Environment(configs=cfg)
robot = env.robots[0]

og.sim.step()

# Make sure none of the joints are moving
robot.keep_still()

for _ in range(50): 
    og.sim.step()
breakpoint()

og.shutdown()
