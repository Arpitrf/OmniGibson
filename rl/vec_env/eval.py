import argparse
import logging
import os
import socket
import sys
from collections import defaultdict

import numpy as np
import yaml

log = logging.getLogger(__name__)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import torch as th
import torch.nn as nn
import wandb
from service.telegym import GRPCClientVecEnv
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecVideoRecorder
from wandb import AlertLevel
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Absolute path to desired PPO checkpoint to load for evaluation",
    required=True,
)
args = parser.parse_args()

def _get_env_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "rl.yaml"))
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config

NUM_EVAL_EPISODES = 10
reset_poses_path = os.path.dirname(__file__) + "/../reset_poses.json"

def train():

    import omnigibson as og
    from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment
    from omnigibson.macros import gm

    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = False

    # Decide whether to use a local environment or remote
    n_envs = 1
    env_config = _get_env_config()
    env_config["task"]["precached_reset_pose_path"] = reset_poses_path
    del env_config["env"]["external_sensors"]
    env = SB3VectorEnvironment(n_envs, env_config, render_on_step=True)
    env = VecFrameStack(env, n_stack=5)
    env = VecMonitor(env, info_keywords=("is_success",))

    prefix = ""
    seed = 0
    # if "dist_coeff" in wandb.config:
    #     task_config = _get_env_config()["task"]
    #     task_config["precached_reset_pose_path"] = reset_poses_path
    #     task_config["reward_config"]["dist_coeff"] = wandb.config.dist_coeff
    #     task_config["reward_config"]["grasp_reward"] = wandb.config.grasp_reward
    #     task_config["reward_config"]["collision_penalty"] = wandb.config.collision_penalty
    #     task_config["reward_config"]["eef_position_penalty_coef"] = wandb.config.eef_position_penalty_coef
    #     task_config["reward_config"]["eef_orientation_penalty_coef"] = (
    #         wandb.config.eef_orientation_penalty_coef_relative * wandb.config.eef_position_penalty_coef
    #     )
    #     task_config["reward_config"]["regularization_coef"] = wandb.config.regularization_coef
    #     env.env_method("update_task", task_config)
    #     eval_env.env_method("update_task", task_config)

    # Set the set
    set_random_seed(seed)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )
    # Need to enable rendering in simulator.step and something else
    # ceiling = env.scene.object_registry("name", "ceilings")
    # ceiling.visible = False
    model = PPO.load(args.checkpoint)
    print("Starting evaluation...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=NUM_EVAL_EPISODES)
    print("Finished evaluation!")
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    train()
