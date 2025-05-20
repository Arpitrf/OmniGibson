import os
import yaml
import  pdb
import pickle
import h5py

import numpy as np
np.set_printoptions(suppress=True, precision=3)
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
from scipy.linalg import expm

from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.arpit_trial.utils.memory import Memory
from omnigibson.utils.python_utils import nums2array
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line, clear_debug_drawing
from omnigibson import object_states

def set_gripper_friction(friction_val=4.0):
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=friction_val,
        dynamic_friction=friction_val,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)
    og.sim.play()
    og.sim.load_state(state)

def move_primitive(robot, action_traj, episode_memory=None, writer=None):
    for action_wrt_world in action_traj:

        print("action_wrt_world: ", action_wrt_world[3:6])

        # convert the action from world frame to robot frame
        robot_pose = robot.get_position_orientation()
        robot_pose = T.pose2mat(robot_pose)
        robot_pose[:3, 3] = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
        homo_action_wrt_world = th.eye(4)
        homo_action_wrt_world[:3, :3] = th.tensor(R.from_rotvec(action_wrt_world[6:9]).as_matrix(), dtype=th.float32)
        homo_action_wrt_world[:3, 3] = th.tensor(action_wrt_world[3:6], dtype=th.float32)
        homo_action_wrt_robot = th.linalg.inv(robot_pose) @ homo_action_wrt_world
        action_wrt_robot = np.concatenate((action_wrt_world[:3], homo_action_wrt_robot[:3, 3], np.array(R.from_matrix(homo_action_wrt_robot[:3, :3]).as_rotvec()), action_wrt_world[-1:]))

        if episode_memory is not None:
            episode_memory.add_action('actions', action_wrt_robot)

        current_pose = robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = action_wrt_robot[3:6]
        print("delta_pos: ", delta_pos)
        delta_orn = action_wrt_robot[6:9]
        # negating the action here for the robotiq gripper as -1 is open and 1 is close
        grasp_action = -action_wrt_robot[9]
        
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
                                                                        grasp_action, 
                                                                        episode_memory,
                                                                        check_grasp=True,
                                                                        writer=writer,
                                                                        log=True)


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

def compute_trajectory_screw(T0, s_hat, q, len_hand_pts, theta_step=0.0):
    computed_Ts = []
    w = s_hat
    v = -np.cross(s_hat, q)
    twist = np.concatenate((w,v)) 
    # Calculate the matrix form of the twist vector
    w = twist[:3]
    w_matrix = [
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ]
    # print("w_matrix: ", w_matrix)
    S = [
        [w_matrix[0][0], w_matrix[0][1], w_matrix[0][2], twist[3]],
        [w_matrix[1][0], w_matrix[1][1], w_matrix[1][2], twist[4]],
        [w_matrix[2][0], w_matrix[2][1], w_matrix[2][2], twist[5]],
        [0, 0, 0, 0]
    ]
    computed_Ts.append(T0)

    # calculate the thetas
    thetas = []
    for i in range(1, len_hand_pts):
        thetas.append(i*theta_step)
    
    # Calculate the transformation of the point when moved by theta along the screw axis
    delta_actions = []
    for j, theta in enumerate(thetas):
        S_theta = theta * np.array(S)
        
        T1 = np.dot(expm(S_theta), T0)

        # obtain the delta pose between T1 and commputed_Ts[-1]
        R_delta = np.dot(T1[:3, :3], computed_Ts[-1][:3, :3].T)
        R_delta_rotvec = R.from_matrix(R_delta).as_rotvec()
        translation_delta = T1[:3, 3] - computed_Ts[-1][:3, 3]
        delta_action = np.concatenate((np.array([0.0, 0.0, 0.0]), translation_delta, R_delta_rotvec, np.array([-1.0])))
        print("delta_action: ", j, delta_action)
        delta_actions.append(delta_action)

        # print('waypoint_pos: ', T1[:3,3])
        computed_Ts.append(T1)
    
    return computed_Ts, delta_actions


# GLOBAL_TIMESTEP = 0.0
# # Create a figure and axis object
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'b-', animated=True)

# # ax.set_xlim(-1, max(xdata) + 1)  # Update x limit based on data
# ax.set_ylim(-1, 20)  # Update y limit based on data

# # Initialize the plot limits and labels
# def init():
#     ax.set_xlim(0, 10)  # Adjust as needed
#     ax.set_ylim(0, 20)  # Adjust as needed
#     # ax.set_ylim(-10, 10)  # Adjust as needed
#     return ln,

# # Update function for the animation
# def update(force_value):

#     print("global timestep: ", GLOBAL_TIMESTEP, force_value)
#     # breakpoint()
#     xdata.append(GLOBAL_TIMESTEP)
#     # ydata.append(force_value.cpu().item())
#     ydata.append(10)
#     ln.set_data(xdata, ydata)
#     plt.draw()
#     plt.pause(0.01)

#     # Dynamically adjust x-axis to accommodate new time steps
#     ax.set_xlim(0, GLOBAL_TIMESTEP + 1)
    
#     # # Adjust limits if needed
#     # if frame >= ax.get_xlim()[1]:
#     #     ax.set_xlim(frame - 10, frame)  # Sliding window

#     # if len(ydata) > 1:
#     #     ax.set_ylim(min(ydata) - 1, max(ydata) + 1)  # Adjust y-axis limits
    
#     return ln,

def dump_to_memory(env, robot, episode_memory):
    obs, obs_info = env.get_obs()

    proprio = robot._get_proprioception_dict()
    # add eef pose and base pose to proprio
    proprio['left_eef_pos'], proprio['left_eef_orn'] = robot.get_relative_eef_pose(arm='left')
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
    
    base_yaw = 0.0
    # base_yaw = np.random.uniform(-20, 20)
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # Randomizing base pos
    base_pos = np.array([0.55, 0.0, 0.0])
    # base_pos = np.array([0.71, -0.05, 0.0])
    # base_pos = np.array([ 0.67, -0.10,  0.0]) #0.72 -0.043, 0.0
    # base_pos = np.array([0.67, 0.0, 0.0])
    # base_x_noise = np.random.uniform(-0.2, 0.1)
    # base_y_noise = np.random.uniform(-0.1, 0.1)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    # print("base_pos: ", base_pos)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.603, -0.897])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

    # add to memory
    if episode_memory is not None:
        dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None, check_grasp=False, writer=None, log=False):
    global GLOBAL_TIMESTEP 
    obs, info = env.get_obs()
    total_collisions = 0
    singularities = []
    reached_singularity = False
    for action in ctrl_gen:
        if action == 'Done':
            print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(action_primitives.move_hand_direct_ik_orn_error))
            normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
            close_to_one = th.isclose(normalized_qpos[:-3], th.tensor(1.0), atol=1e-2)
            close_to_neg_one = th.isclose(normalized_qpos[:-3], th.tensor(-1.0), atol=1e-2)
            any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
            # print("any_close_to_one_or_neg_one: ", any_close_to_one_or_neg_one)
            if any_close_to_one_or_neg_one:
                print("Reached joint limits. Exiting", normalized_qpos)
                # remove the last action from memory
                if episode_memory is not None:
                    episode_memory.data['actions']['actions'].pop()
                return False

            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 

            # Sidestep control issues. If pose error gets large, means the previous action was bad and so we save that action and stop the episode
            pos_thresh = 0.04
            ori_thresh = 0.1
            reached_goal = action_primitives.move_hand_direct_ik_pos_error < pos_thresh and action_primitives.move_hand_direct_ik_orn_error < ori_thresh
            if not reached_goal:
                print("Did not reach waypoint. Exiting. Normalized_qpos: ", robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]])
                return False

            continue
        
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
        # GLOBAL_TIMESTEP += 1
        
        # # force_data_generator()
        # force_value = robot.get_joint_forces()
        # update(force_value)
        
        # if singularity is reached in this action, do not add to memory
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)

        if sum(singularities) > 10:
            reached_singularity = True
            print("Reached singularity!")
            # remove the last action from memory
            # episode_memory.data['actions']['actions'].pop()
            return False
        
        # Check grasp
        is_grasping = robot.custom_is_grasping()
        if check_grasp and not is_grasping:
            print("Grasp failed. Exiting.", robot._get_proprioception_dict()['gripper_right_qpos'])
            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            return False

    return True

def primitive(episode_memory=None, episode_number=0):
    grasp_action = -1.0
    # # ======================= Move base ================================  
    # # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
    # target_base_pose = th.tensor([0.0, 0.0, 1.57])
    # execute_controller(action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory)    
    # # for _ in range(50):
    # #     og.sim.step()
    # curr_base_pos = robot.get_position()
    # print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
    # # =================================================================================

    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    # # w.r.t world robotiq gripper (side grasp)
    # target_pose_world = th.tensor([
    #     [ 0.93739022,  0.01530303,  0.34794453,  1.26256945], #1.24256945
    #     [-0.34824335,  0.02651403,  0.93702912, -0.15659404],
    #     [ 0.00511398, -0.9995313,   0.03018317,  0.50507379],
    #     [ 0.        ,  0.,          0.,          1.        ],
    # ])
    
    # w.r.t world robotiq gripper (front grasp)
    target_pose_world = th.tensor([
        [-0.578,  0.015,  0.816,  1.017],
        [-0.815, -0.059, -0.576, -0.875],
        [ 0.04 , -0.998,  0.047,  0.666],
        [ 0.   ,  0.,     0.,     1.   ],
    ])

    target_quat = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (target_pose_world[:3, 3], th.tensor(target_quat, dtype=th.float32))
    # target_pose_world = T.mat2pose(target_pose_world)
    
    pre_target_pose = (target_pose_world[0] + th.tensor([-0.04, 0.04, 0.0]), target_pose_world[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory) 
    
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

    target_joint_pos = robot.get_joint_positions(normalized=False)[robot.arm_control_idx["right"]]
    # ============= Perform grasp ===================
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(20):
        og.sim.step()
        # # print joint error
        # current_joint_pos = robot.get_joint_positions(normalized=False)[robot.arm_control_idx["right"]]
        # joint_error = np.linalg.norm(target_joint_pos - current_joint_pos)
        # print("joint_error: ", joint_error)
    # ==============================================

    # right_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    # right_eef_pose_world = T.pose2mat(right_eef_pose_world)
    # axis = np.array([0.0, 0.0, 1.0])
    # q = np.array([1.35, -0.38,  0.1])

    # # visualize the axis
    # prev_position = q
    # next_position = prev_position + np.array([0.0, 0.0, 0.5])
    # draw_line(prev_position, next_position, size=5.0)
    # # breakpoint()

    # len_hand_pts = 6

    # computed_Ts_screw, delta_actions = compute_trajectory_screw(right_eef_pose_world.numpy(), axis, q, len_hand_pts, theta_step=0.3)
    # breakpoint()

    # move_primitive(robot, delta_actions, episode_memory=episode_memory)

    # for waypoint in computed_Ts_screw:
    #     # breakpoint()
    #     waypoint = th.tensor(waypoint, dtype=th.float32)
    #     target_pose = T.mat2pose(waypoint)
    #     action_exec = execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory,
    #                    check_grasp=True)
    #     if not action_exec:
    #         break 

    breakpoint()
    # # ======================= Move hand back ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    new_pos = curr_pos + th.tensor([-0.3, -0.2, 0.0])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory,
                       check_grasp=True)
    
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory)
    
    # for _ in range(40):
    #     og.sim.step()
    
    # Debugging
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # # # =================================================================================

    # # ============= Open grasp =================
    # grasp_action = -1.0
    # action = action_primitives._empty_action()
    # action[robot.gripper_action_idx["right"]] = grasp_action
    # env.step(action)
    # for _ in range(40):
    #     og.sim.step()
    # # # save everything to memory
    # # dump_to_memory(env, robot, episode_memory)
    # # # TODO: Change the indexing here
    # # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # # episode_memory.add_action('actions', action_to_add)
    # # # ==========================================

    for _ in range(50):
        og.sim.step()



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# robot specific config
config["robots"][0]["default_arm_pose"] = "horizontal"
config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
config["robots"][0]["controller_config"]["arm_right"]["kp"] = 50.0
# config["robots"][0]["controller_config"]["arm_right"]["kp"] = th.tensor([500.0, 500.0, 500.0, 500.0, 250.0, 250.0, 250.0])
config["robots"][0]["controller_config"]["gripper_right"]["motor_type"] = "velocity"

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -90.0]
# for hivvdf (upside down)
# rot_euler = [180.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
obj_cfg = dict(
    type="DatasetObject",
    name="oven",
    category="oven",
    model="leqtlc", #petcxr
    position=[1.5, -0.6, 1.0],
    scale=[1.5, 1.0, 1.0],
    orientation=rot_quat,
    )
coffe_table_cfg = dict(
    type="DatasetObject",
    name="coffee_table",
    category="coffee_table",
    model="fqluyq", #petcxr
    # position=[0, 0.6, 0.3],
    position=[1.5, -0.6, 0.4],
    scale=[2.0, 2.0, 1.0],
    orientation=rot_quat,
)
config["objects"] = [obj_cfg, coffe_table_cfg]

# oven.states[object_states.OnTop].set_value(other=coffee_table, new_value=True)

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

set_gripper_friction(friction_val=50.0)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-0.7563,  1.1324,  1.0464]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)
for _ in range(2): og.sim.step()


# # setting properties of the objects
object_name = "coffee_table"
coffee_table = env.scene.object_registry("name", object_name)
coffee_table.root_link.mass = 1000.0
coffee_table.keep_still()

object_name = "oven"
oven = env.scene.object_registry("name", object_name)
oven.root_link.mass = 200.0
oven.joints['j_dof_rootd_aa001_r'].friction = 1000.0

for _ in range(20): og.sim.step()
breakpoint()

receptacle_joint_pos = -1.0
oven.joints["j_dof_rootd_aa001_r"].set_pos(receptacle_joint_pos, normalized=True)
coffee_table.keep_still()
oven.keep_still()

state = og.sim.dump_state(serialized=False)
for _ in range(1):
    custom_reset(env, robot)
    coffee_table.keep_still()
    oven.keep_still()
    breakpoint()
    # primitive()
    # episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # og.sim.load_state(state, serialized=False)
    # for _ in range(10):
    #     og.sim.step()

    # episode_number += 1

# =============================== Teleop ===============================
# set control gains for easier teleop
robot.controllers["arm_right"].kp = 50.0
# robot.controllers["gripper_right"]._motor_type = "velocity"
for _ in range(50): og.sim.step()

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

max_steps = -1 
step = 0
breakpoint()
while step != max_steps:
    action, keypress_str = action_generator.get_teleop_action()
    # action[robot.gripper_action_idx["right"]] = 1.0
    # print("action: ", action)
    env.step(action=action)
    if keypress_str == 'TAB':
        right_eef_pos_world, right_eef_orn_world = robot.eef_links["right"].get_position_orientation()
        right_eef_pose_world = np.eye(4)
        right_eef_pose_world[:3, :3] = R.from_quat(right_eef_orn_world).as_matrix()
        right_eef_pose_world[:3, 3] = right_eef_pos_world

        obj_pos_world, obj_orn_world = scene.object_registry("name", object_name).get_position_orientation()
        obj_pose_world = np.eye(4)
        obj_pose_world[:3, :3] = R.from_quat(obj_orn_world).as_matrix()
        obj_pose_world[:3, 3] = obj_pos_world

        if np.linalg.det(obj_pose_world) != 0:
            right_eef_pose_object = np.linalg.inv(obj_pose_world) @ right_eef_pose_world
        else:
            right_eef_pose_object = np.linalg.pinv(obj_pose_world) @ right_eef_pose_world

        base_pose = robot.get_position_orientation()
        right_eef_pose = robot.get_relative_eef_pose(arm='right')
        print("right_eef_pose: ", right_eef_pose)
        print("right_eef_pose_world: ", right_eef_pose_world)
        print("right_eef_pose_object: ", right_eef_pose_object)
        print("base_pose: ", base_pose)
        breakpoint()

    step += 1
# ========================================================================

breakpoint()

# Always shut down the environment cleanly at the end
# og.clear()