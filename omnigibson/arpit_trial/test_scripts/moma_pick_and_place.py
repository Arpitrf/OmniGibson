import os
import yaml
import  pdb
import pickle
import h5py

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.arpit_trial.utils.memory import Memory

from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator


def set_gripper_joint_positions(robot, q, attached_obj):
    # Overwrite the gripper joint positions based on attached_obj
    joint_names = list(robot.joints.keys())
    for arm, finger_joints in robot.finger_joints.items():
        close_gripper = attached_obj is not None and robot.eef_link_names[arm] in attached_obj
        for finger_joint in finger_joints:
            idx = joint_names.index(finger_joint.joint_name)
            q[idx] = finger_joint.lower_limit if close_gripper else finger_joint.upper_limit
    return q


def control_gripper(env, robot, attached_obj):
    # Control the gripper to open or close, while keeping the rest of the robot still
    q = robot.get_joint_positions()
    q = set_gripper_joint_positions(robot, q, attached_obj)
    command = q_to_command(q, robot)
    num_repeat = 30
    print(f"Gripper (attached_obj={attached_obj})")
    for _ in range(num_repeat):
        env.step(command)


def q_to_command(q, robot):
    # Convert target joint positions to command
    command = []
    
    # remove later
    q[robot.base_control_idx] = th.tensor([0.0, 0.0, 0.0])

    for controller in robot.controllers.values():
        command.append(q[controller.dof_idx])
    command = th.cat(command, dim=0)
    assert command.shape[0] == robot.action_dim
    return command

def plan_trajectory(
    cmg, target_pos, target_quat, emb_sel=CuroboEmbodimentSelection.DEFAULT, attached_obj=None, attached_obj_scale=None, is_local=False, ignore_objs_in_collision_check=None
):
    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
    successes, traj_paths = cmg.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        is_local=is_local,
        max_attempts=50,
        timeout=60.0,
        ik_fail_return=5,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=1.0,
        attached_obj=attached_obj,
        attached_obj_scale=attached_obj_scale,
        emb_sel=emb_sel,
        ignore_objs_in_collision_check=ignore_objs_in_collision_check
    )
    return successes, traj_paths


def execute_trajectory(q_traj, env, robot, attached_obj):
    for i, q in enumerate(q_traj):
        q = q.cpu()
        q = set_gripper_joint_positions(robot, q, attached_obj)

        # breakpoint()
        command = q_to_command(q, robot)

        num_repeat = 5
        print(f"Executing waypoint {i}/{len(q_traj)}")
        for _ in range(num_repeat):
            robot.controllers["base"]._goal["target"] = th.tensor([0.0, 0.0, 0.0])
            env.step(command)


def plan_and_execute_trajectory(
    cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot, dry_run=False, is_local=False, ignore_objs_in_collision_check=None
):
    successes, traj_paths = plan_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, is_local=is_local, ignore_objs_in_collision_check=ignore_objs_in_collision_check)
    success, traj_path = successes[0], traj_paths[0]
    # Move the markers to the desired eef positions
    for marker in eef_markers:
        marker.set_position_orientation(position=th.tensor([100, 100, 100]))
    for target_link, marker in zip(target_pos.keys(), eef_markers):
        marker.set_position_orientation(position=target_pos[target_link])
    if success:
        print("Successfully planned trajectory")
        if not dry_run:
            q_traj = cmg.path_to_joint_trajectory(traj_path, emb_sel)
            execute_trajectory(q_traj, env, robot, attached_obj)
    else:
        print("Failed to plan trajectory")

def dump_to_memory(env, robot, episode_memory):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    # proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
    proprio['right_eef_pos'], proprio['right_eef_orn'] = robot.get_relative_eef_pose(arm='right')
    proprio['base_pos'], proprio['base_orn'] = robot.get_position_orientation()
    for k in proprio.keys():
        episode_memory.add_proprioception(k, proprio[k].cpu().numpy())

    robot_name = env.robots[0].name
    for k in obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation(k, obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k].cpu().numpy())
    # # add gripper+object seg
    # gripper_obj_seg = obtain_gripper_obj_seg(obs[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'], obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0']['seg_instance_id'])
    # episode_memory.add_observation('gripper_obj_seg', gripper_obj_seg)
    
    for k in obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'].keys():
        episode_memory.add_observation_info(k, obs_info[f'{robot_name}'][f'{robot_name}:eyes:Camera:0'][k])


    is_grasping = robot.custom_is_grasping()
    is_contact = detect_robot_collision_in_sim(robot)

    episode_memory.add_extra('grasps', is_grasping.numpy())
    episode_memory.add_extra('contacts', is_contact)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # randomizing base pos
    base_pos = np.array([-0.05, -0.4, 0.0])
    base_x_noise = np.random.uniform(-0.15, 0.15)
    base_y_noise = np.random.uniform(-0.15, 0.15)
    base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    base_pos += base_noise 
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

    # add to memory
    # dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    obs, info = env.get_obs()
    total_collisions = 0
    for action in ctrl_gen:
        if action == 'Done':
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)
    return obs, info, total_collisions

def curobo_primitive():
    # ======================= Move base ================================  
    grasp_action = 1.0
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.0, 0.0, 1.57])
    execute_controller(action_primitives._navigate_to_pose_direct(target_base_pose), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)    
    # for _ in range(50):
    #     og.sim.step()
    curr_base_pos = robot.get_position()
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    # =================================================================================

    # emb_sel = CuroboEmbodimentSelection.BASE
    # attached_obj = None
    # attached_obj_scale = None
    # # target_base_pose = th.tensor([0.0, 0.0, 1.57])
    # target_nav_pos = th.tensor([0.0, 0.0, 0.0])
    # target_nav_quat = th.tensor(R.from_euler('z', 90, degrees=True).as_quat())
    # target_pos = {robot.base_footprint_link_name: target_nav_pos}
    # target_quat = {robot.base_footprint_link_name: target_nav_quat}
    # plan_and_execute_trajectory(
    #     cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    # )

    # right_hand_pos, right_hand_quat = robot.get_eef_pose(arm="right")
    target_pose = (th.tensor([0.1235, 0.4631, 0.4138]), th.tensor([-0.0091,  0.0041,  0.9619,  0.2733]))
    pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
    
    emb_sel = CuroboEmbodimentSelection.ARM
    attached_obj = None
    attached_obj_scale = None
    ignore_objs_in_collision_check = ["/World/scene_0/box"]
    
    target_pos = {robot.eef_link_names["right"]: pre_target_pose[0]}
    target_quat = {robot.eef_link_names["right"]: pre_target_pose[1]}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot, ignore_objs_in_collision_check=ignore_objs_in_collision_check
    )

    target_pos = {robot.eef_link_names["right"]: target_pose[0]}
    target_quat = {robot.eef_link_names["right"]: target_pose[1]}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot, ignore_objs_in_collision_check=ignore_objs_in_collision_check
    )

    box = env.scene.object_registry("name", "box")
    attached_obj = {robot.eef_link_names["right"]: box.root_link}
    control_gripper(env, robot, attached_obj)

    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(40):
    #     og.sim.step()


    # move hand up
    # curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    curr_pos, curr_orn = robot.get_eef_pose(arm='right')
    breakpoint()
    # curr_orn = th.tensor([-0.0091,  0.0041,  0.9619,  0.2733])
    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.1])
    target_pose = (new_pos, curr_orn)
    target_pos = {robot.eef_link_names["right"]: target_pose[0]}
    target_quat = {robot.eef_link_names["right"]: target_pose[1]}
    attached_obj = {robot.eef_link_names["right"]: box.root_link}
    attached_obj_scale = {robot.eef_link_names["right"]: 0.99}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )


def primitive(episode_memory=None, episode_number=0):

    # ======================= Move base ================================  
    grasp_action = 1.0
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.0, 0.0, 1.57])
    execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)    
    # for _ in range(50):
    #     og.sim.step()
    curr_base_pos = robot.get_position()
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    # =================================================================================

    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # w.r.t robot
    # target_pose:  (th.tensor([ 0.4891, -0.1747,  0.3917]), th.tensor([-0.0224,  0.0234,  0.6525,  0.7571]))
    # w.r.t world
    # target_pose = (th.tensor([0.1747, 0.4891, 0.3922]), th.tensor([-3.2351e-02,  6.7136e-04,  9.9674e-01,  7.3904e-02]))
    # # diagonal 45
    # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
    # vertical
    # w.r.t robot
    # target_pose = (th.tensor([ 0.5066, -0.0575,  0.4948]), th.tensor([ 0.4775,  0.5259, -0.5041,  0.4913]))
    # w.r.t world
    # target_pose = (th.tensor([0.0933, 0.5011, 0.4953]), th.tensor([ 0.0090, -0.7102,  0.0341, -0.7031]))
    # horizontal-forward
    # w.r.t world
    target_pose = (th.tensor([0.1235, 0.4831, 0.4138]), th.tensor([-0.0091,  0.0041,  0.9619,  0.2733]))

    pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory) 
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

    # ============= Perform grasp ===================
    grasp_action = -1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    dump_to_memory(env, robot, episode_memory)
    # TODO: Change the indexing here
    action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    episode_memory.add_action('actions', action_to_add)
    # ==============================================
        
    # ======================= Move hand up ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    new_pos = curr_pos + th.tensor([0.0, 0.0, 0.2])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)
    
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

    # ============= Move base ===================
    # debugging
    ee_pose_before_nav = robot.get_relative_eef_pose(arm='right')
    # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    target_base_pose = th.tensor([0.456, 0.0257, 0.0]) # [0.526, 0.0257, 0.0]
    execute_controller(action_primitives._navigate_to_pose_direct(target_base_pose),
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)    
    for _ in range(50):
        og.sim.step()
    curr_base_pos = robot.get_position_orientation()[0]
    print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    
    # Debugging
    ee_pose_after_nav = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(ee_pose_after_nav[0] - ee_pose_before_nav[0])
    orn_error = T.get_orientation_diff_in_radian(ee_pose_after_nav[1], ee_pose_before_nav[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # ============================================
    
    og.sim.save([f'{save_folder}/episode_{episode_number:05d}_place_start.json'])

    # ======================= Move hand to place pose ================================
    # w.r.t world
    # place_pose =  (th.tensor([ 1.10402, -0.1873,  0.8563]), th.tensor([-0.0488, -0.0116,  0.5546,  0.8306])) # [ 1.1602, -0.1873,  0.8463]
    # w.r.t robot
    # place_pose = (th.tensor([0.6458, -0.2320, 0.8481]), th.tensor([-0.0555, -0.0157, 0.5436, 0.8373]))

    # for vertical
    # w.r.t world
    place_pose =  (th.tensor([ 1.10402, -0.1873,  0.9563]), th.tensor([ 0.0090, -0.7102,  0.0341, -0.7031]))
    execute_controller(action_primitives._move_hand_linearly_cartesian(place_pose, ignore_failure=True, in_world_frame=True, episode_memory=episode_memory, grasp_action=grasp_action), 
                       env, 
                       robot, 
                       grasp_action,
                       episode_memory)
    # execute_controller(action_primitives._move_hand_direct_ik(place_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action)
    current_pose_world = robot.eef_links["right"].get_position_orientation()
    print("move hand to place location completed. Desired and Reached right eef pose reached: ", place_pose[0], current_pose_world)
    # input()
    # ====================================================================================

    # ============= Open grasp =================
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # save everything to memory
    dump_to_memory(env, robot, episode_memory)
    # TODO: Change the indexing here
    action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    episode_memory.add_action('actions', action_to_add)
    # ==========================================

    for _ in range(50):
        og.sim.step()



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'absolute_pose'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, -30.0]
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
        # "visual_only": True,
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        # "size": 0.05,
        "mass": 1e-6,
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
    {
        "type": "PrimitiveObject",
        "name": "eef_marker_0",
        "primitive_type": "Sphere",
        "radius": 0.05,
        "visual_only": True,
        "position": [0, 0, 0],
        "orientation": [0, 0, 0, 1],
        "rgba": [1, 0, 0, 1],
    },
    {
        "type": "PrimitiveObject",
        "name": "eef_marker_1",
        "primitive_type": "Sphere",
        "radius": 0.05,
        "visual_only": True,
        "position": [0, 0, 0],
        "orientation": [0, 0, 0, 1],
        "rgba": [0, 1, 0, 1],
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
print(robot.name)

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=200.0,
    dynamic_friction=200.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)

og.sim.play()
og.sim.load_state(state)

action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# pdb.set_trace()
# obj = env.scene.object_registry("name", "box")
# obj.root_link.mass = 1e-2
# print("obj.mass: ", obj.mass)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

for _ in range(20):
    og.sim.step()

save_folder = 'moma_pick_and_place'
os.makedirs(save_folder, exist_ok=True)
episode_memory = Memory()

episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
        episode_number = len(file['data'].keys())
        print("episode_number: ", episode_number)

# Create CuRobo instance
cmg = CuRoboMotionGenerator(
    robot=robot,
    batch_size=1,
    use_cuda_graph=True,
)
eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]

for _ in range(1):
    # custom_reset(env, robot, episode_memory)
    # # save the start simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    # primitive(episode_memory, episode_number)

    curobo_primitive()

    # episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # # save the end simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    episode_number += 1

# # =============================== Teleop ===============================
# # Create teleop controller
# action_generator = KeyboardRobotController(robot=robot)
# # Register custom binding to reset the environment
# action_generator.register_custom_keymapping(
#     key=lazy.carb.input.KeyboardInput.R,
#     description="Reset the robot",
#     callback_fn=lambda: env.reset(),
# )
# # Print out relevant keyboard info if using keyboard teleop
# action_generator.print_keyboard_teleop_info()

# max_steps = -1 
# step = 0
# while step != max_steps:
#     action, keypress_str = action_generator.get_teleop_action()
#     print("action: ", action)
    
#     # if action = SPECIAL_ACTION / NONE:
#     #     do not do pre_step()
#     # og.sim.render()
#     # if any(action[robot.controller_action_idx["base"]] != 0.0) or \
#     #         any(action[robot.controller_action_idx["camera"]] != 0.0) or \
#     #         any(action[robot.controller_action_idx["arm_left"]] != 0.0) or \
#     #         any(action[robot.controller_action_idx["arm_right"]] != 0.0):

#     env.step(action=action)
#     if keypress_str == 'TAB':
#         right_eef_pose = robot.get_relative_eef_pose(arm='right')
#         right_eef_pose_world = robot.eef_links["right"].get_position_orientation()
#         base_pose = robot.get_position_orientation()
#         print("right_eef_pose: ", right_eef_pose)
#         print("right_eef_pose_world: ", right_eef_pose_world)
#         print("base_pose: ", base_pose)
#         breakpoint()
#         # og.sim.save([f'temp2.json'])
#     step += 1
# # ========================================================================

breakpoint()
for _ in range(500):
    og.sim.step()

# Always shut down the environment cleanly at the end
# og.clear()