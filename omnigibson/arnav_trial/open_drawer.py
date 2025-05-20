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
from omnigibson.utils.python_utils import nums2array


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
    
    base_yaw = -10
    # base_yaw = np.random.uniform(-20, 20)
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # Randomizing base pos
    # base_pos = np.array([0.55, 0.0, 0.0])
    base_pos = np.array([-0.1, 0.0, 0.0])
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
    dump_to_memory(env, robot, episode_memory)

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    global GLOBAL_TIMESTEP 
    obs, info = env.get_obs()
    total_collisions = 0
    singularities = []
    reached_singularity = False
    for action in ctrl_gen:
        if action == 'Done':
            print("pos and orn errors: ", action_primitives.move_hand_direct_ik_pos_error, th.rad2deg(action_primitives.move_hand_direct_ik_orn_error))
            normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
            print("normalized_qpos: ", normalized_qpos)
            close_to_one = th.isclose(normalized_qpos[:-3], th.tensor(1.0), atol=1e-2)
            close_to_neg_one = th.isclose(normalized_qpos[:-3], th.tensor(-1.0), atol=1e-2)
            any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
            print("any_close_to_one_or_neg_one: ", any_close_to_one_or_neg_one)
            # if any_close_to_one_or_neg_one:
            #     safe = False
            #     unsafe_reasons.append("Reaching some joint limit") 

            if episode_memory is not None:
                dump_to_memory(env, robot, episode_memory) 
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # GLOBAL_TIMESTEP += 1
        
        # # force_data_generator()
        # force_value = robot.get_joint_forces()
        # update(force_value)
        
        # if singularity is reached in this action, do not add to memory
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)

        if sum(singularities) > 3:
            reached_singularity = True
            print("Reached singularity!")
            breakpoint()
            # remove the last action from memory
            # episode_memory.data['actions']['actions'].pop()
            return reached_singularity

    return obs, info, total_collisions

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
    # # w.r.t object
    # target_pose_obj = np.array([
    #     [-0.21300013, -0.03880069,  0.96416772,  0.04917458],
    #     [ 0.36973975, -0.11093961, -0.06906067,  0.2582519 ],
    #     [-0.25799225,  0.94084111, -0.05303142,  0.44741508],
    #     [ 0.41167921,  0.15094028,  0.11901726,  0.56770111],
    # ])
    # w.r.t object robotiq gripper
    target_pose_obj = np.array([
        [-0.03007303, -0.98993309, -0.13830437,  0.03215751],
        [-0.16730912, -0.13142948,  0.97710488, -0.3610887 ],
        [-0.98544572,  0.05252409, -0.16167235,  0.30995518],
        [ 0.        ,  0.,          0.,          1.        ],
    ])
    # # w.r.t world robotiq gripper
    # target_pose_world = th.tensor([
    #     [-0.16730904, -0.13142948,  0.9771049 ,  1.10891759],
    #     [ 0.03007308,  0.98993308,  0.13830438, -0.28215751],
    #     [-0.98544573,  0.05252413, -0.16167226,  0.65757006],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ],
    # ])
    # target_pose_world = T.mat2pose(target_pose_world)

    cabinet_pos_world, cabinet_orn_world = scene.object_registry("name", "drawer").get_position_orientation()
    cabinet_pose_world = np.eye(4)
    cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    cabinet_pose_world[:3, 3] = cabinet_pos_world
    target_pose_world = cabinet_pose_world @ target_pose_obj
    target_pos_world = target_pose_world[:3, 3]
    target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (th.tensor(target_pos_world, dtype=th.float32), th.tensor(target_orn_world, dtype=th.float32))
    

    

    # pre_target_pose = (target_pose_world[0] + th.tensor([0.0, 0.0, 0.1]), target_pose_world[1]) 
    # execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory) 
    
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose_world, ignore_failure=True, in_world_frame=True), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose_world[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose_world[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================
    breakpoint()

    # ============= Perform grasp ===================
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
    # ==============================================
        
    # # ======================= Move hand back ================================  
    curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
    new_pos = curr_pos + th.tensor([-0.3, 0.0, 0.0])
    target_pose = (new_pos, curr_orn)
    execute_controller(action_primitives._move_hand_linearly_cartesian(target_pose, ignore_failure=True, in_world_frame=False), 
                       env, 
                       robot, 
                       grasp_action, 
                       episode_memory)
    
    # execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory)
    
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # # =================================================================================

    # ============= Open grasp =================
    grasp_action = -1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # # save everything to memory
    # dump_to_memory(env, robot, episode_memory)
    # # TODO: Change the indexing here
    # action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
    # episode_memory.add_action('actions', action_to_add)
    # # ==========================================

    for _ in range(50):
        og.sim.step()



config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0
config["robots"][0]["controller_config"]["gripper_right"]["motor_type"] = "velocity"

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, -90.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
# obj_cfg = dict(
#     type="DatasetObject",
#     name="fridge",
#     category="fridge",
#     model="hivvdf",
#     position=[1.5, -0.6, 1.0],
#     # scale=[2.0, 1.0, 1.0],
#     orientation=rot_quat,
#     )
# obj_cfg = dict(
#     type="DatasetObject",
#     name="drawer",
#     category="bottom_cabinet",
#     # visual_only=True,
#     model="rntwkg",
#     position=[1.5, -0.25, 1.0],
#     scale=[1.0, 2.0, 1.2],
#     orientation=rot_quat,
#     )
# obj_cfg = dict(
#     type="DatasetObject",
#     name="drawer",
#     category="bottom_cabinet",
#     # visual_only=True,
#     model="dsbcxl",
#     position=[1.5, -0.25, 1.0],
#     scale=[1.0, 2.0, 1.2],
#     orientation=rot_quat,
#     )
# 
obj_cfg = dict(
    type="DatasetObject",
    name="drawer",
    category="bottom_cabinet",
    # visual_only=True,
    model="pkdnbu",
    position=[1.5, -0.25, 1.0],
    scale=[1.0, 2.5, 1.5],
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
    static_friction=0.5,
    dynamic_friction=0.5,
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

drawer = env.scene.object_registry("name", "drawer")
drawer.root_link.mass = 50.0
# drawer.links["link_5"].mass = 1000.0
# drawer.links["link_4"].mass = 50.0
drawer.links["link_3"].mass = 2000.0
drawer.links["link_2"].mass = 50.0
drawer.links["link_1"].mass = 50.0
drawer.joints["j_link_3"].friction = 5000.0


# joint_val = drawer.joints["j_link_1"].get_state()[0].item()

for _ in range(20):
    og.sim.step()

save_folder = 'open_drawer_temp'
os.makedirs(save_folder, exist_ok=True)
episode_memory = Memory()

episode_number = 0
if os.path.isfile(f'{save_folder}/dataset.hdf5'):
    with h5py.File(f'{save_folder}/dataset.hdf5', 'r') as file:
        episode_number = len(file['data'].keys())
        print("episode_number: ", episode_number)

# robot.controllers["arm_right"].kp[:3] = nums2array(nums=2000, dim=3, dtype=th.float32)
# robot.controllers["arm_right"].kp[-3:] = nums2array(nums=5000, dim=3, dtype=th.float32)

# setting properties of the objects


# ani = animation.FuncAnimation(fig, update, frames=force_data_generator, init_func=init, blit=True, interval=100)
# plt.show()

# init()
# plt.ion()  # Turn on interactive mode
# plt.show()  # Display the plot


state = og.sim.dump_state(serialized=False)
for _ in range(1):
    custom_reset(env, robot, episode_memory)
    drawer.joints["j_link_3"].set_pos(drawer.joints["j_link_3"].upper_limit - 0.09)
    # # save the start simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_start.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_start.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    # primitive(episode_memory, episode_number)

    # episode_memory.dump(f'{save_folder}/dataset.hdf5')

    # # save the end simulator state
    # og.sim.save([f'{save_folder}/episode_{episode_number:05d}_end.json'])
    # arr = scene.dump_state(serialized=True)
    # with open(f'{save_folder}/episode_{episode_number:05d}_end.pickle', 'wb') as f:
    #     pickle.dump(arr, f)

    # og.sim.load_state(state, serialized=False)
    # for _ in range(10):
    #     og.sim.step()

    # episode_number += 1

# =============================== Teleop ===============================
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
gripper_close = False
tab_pressed = False
while step != max_steps:
    action, keypress_str = action_generator.get_teleop_action()
    if keypress_str != "None" and tab_pressed:
        print(keypress_str, action)

    if keypress_str == 'T':
        gripper_close = not gripper_close
        tab_pressed = True
    if gripper_close:
        action[robot.gripper_action_idx["right"]] = 1.0
    else:
        action[robot.gripper_action_idx["right"]] = -1.0
    # print("action: ", action)
    
    # if action = SPECIAL_ACTION / NONE:
    #     do not do pre_step()
    # og.sim.render()
    # if any(action[robot.controller_action_idx["base"]] != 0.0) or \
    #         any(action[robot.controller_action_idx["camera"]] != 0.0) or \
    #         any(action[robot.controller_action_idx["arm_left"]] != 0.0) or \
    #         any(action[robot.controller_action_idx["arm_right"]] != 0.0):

    env.step(action=action)
    if keypress_str == 'TAB':
        right_eef_pos_world, right_eef_orn_world = robot.eef_links["right"].get_position_orientation()
        right_eef_pose_world = np.eye(4)
        right_eef_pose_world[:3, :3] = R.from_quat(right_eef_orn_world).as_matrix()
        right_eef_pose_world[:3, 3] = right_eef_pos_world

        obj_pos_world, obj_orn_world = scene.object_registry("name", "drawer").get_position_orientation()
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