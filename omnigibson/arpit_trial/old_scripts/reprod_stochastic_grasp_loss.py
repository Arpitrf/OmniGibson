import os
import yaml

import numpy as np
import torch as th
import omnigibson as og

from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives



def move_primitive():
        current_pose = robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = th.tensor([-0.05, -0.01, 0.0])
        delta_orn = th.tensor([0.0, 0.0, 0.0])
        grasp_action = -1.0

        target_pos = current_pos + delta_pos
        target_pos = target_pos.type(th.FloatTensor)
        target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
        target_orn = th.tensor(target_orn.as_quat())
        target_orn = target_orn.type(th.FloatTensor)

        target_pose = (target_pos, target_orn)
        
        action_exec = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                                stop_on_contact=False,
                                                                                ignore_failure=True,
                                                                                stop_if_stuck=False,
                                                                                in_world_frame=False), 
                                                                        env, 
                                                                        robot, 
                                                                        grasp_action,)

        # breakpoint()
        for _ in range(50):
            og.sim.step()

        ee_pose_after = robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
        orn_error = orn_error % (2*th.pi)
        print("prev_pos, target_pos, reached_pos: ", current_pos, target_pos, ee_pose_after[0])
        print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")

        if action_exec is False:
            return 


def custom_reset(env, robot): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 0
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # randomizing base pos
    base_pos = np.array([0.55, 0.0, 0.0])
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()


def execute_controller(ctrl_gen, env, robot, grasp_action):
    for action in ctrl_gen:
        if action == "Done":
            continue           
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)


def grasp_handle(env, robot, episode_memory=None):
    # ======================= Move hand to grasp pose ================================    
    grasp_action = 1.0
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # w.r.t object
    target_pose_obj = np.array([
        [-0.21300013, -0.03880069,  0.96416772,  0.04917458],
        [ 0.36973975, -0.11093961, -0.06906067,  0.2582519 ],
        [-0.25799225,  0.94084111, -0.05303142,  0.44741508],
        [ 0.41167921,  0.15094028,  0.11901726,  0.56770111],
    ])
    cabinet_pos_world, cabinet_orn_world = scene.object_registry("name", "bottom_cabinet").get_position_orientation()
    cabinet_pose_world = np.zeros((4, 4))
    cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    cabinet_pose_world[:3, 3] = cabinet_pos_world
    target_pose_world = cabinet_pose_world @ target_pose_obj
    target_pos_world = target_pose_world[:3, 3]
    target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (th.tensor(target_pos_world), th.tensor(target_orn_world))
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose_world, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose_world[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose_world[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

    # # ============= Perform grasp ===================
    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(100):
    #     og.sim.step()
    # # ==============================================




def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=2)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

robot_cfg = [
     {
        "type": "Tiago",
        "obs_modalities": "rgb",
        "self_collisions": True,
        "action_normalize": False,
        "action_type": "continuous",
        "grasping_mode": "physical",
        "rigid_trunk": "false",
        "default_trunk_offset": 0.15,
        "default_arm_pose": "vertical",
        "controller_config": {
            "base": {
                "name": "JointController",
            },
            "arm_left": {
                "name": "NullJointController",
            },
            "arm_right": {
                "name": "InverseKinematicsController",
            },
            "gripper_left": {
                "name": "MultiFingerGripperController",
            },
            "gripper_right": {
                "name": "MultiFingerGripperController",
            },
        },
    },
]

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
obj_cfg = dict(
    type="DatasetObject",
    name="bottom_cabinet",
    category="bottom_cabinet",
    model="rntwkg",
    position=[1.5, -0.25, 1.0],
    scale=[1.0, 1.0, 1.2],
    orientation=rot_quat,
    )
config["objects"] = [obj_cfg]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)
og.sim.play()
og.sim.load_state(state)

# setting properties of the objects
drawer = env.scene.object_registry("name", "bottom_cabinet")
drawer.links["link_5"].mass = 100.0

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([0.88,  0.76,  0.98]),
    th.tensor([-0.12,  0.50,  0.83, -0.20]),
)

for _ in range(20):
    og.sim.step()

num_samples = 10
# Shortcut: take to grasp pose already
custom_reset(env, robot)
grasp_handle(env=env, robot=robot)

state = og.sim.dump_state(serialized=False)
breakpoint()
for i in range(num_samples):
    print(f"---------------- Episode {i} ------------------")    

    # ============= Perform grasp ===================
    grasp_action = -1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(100):
        og.sim.step()
    # ==============================================
    
    move_primitive()
    og.sim.load_state(state, serialized=False)
    breakpoint()
    
    for _ in range(10):
        og.sim.step()
    # breakpoint()

# Always shut down the environment cleanly at the end
og.shutdown()