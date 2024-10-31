import os
import yaml
import  pdb
import pickle
import cv2
import imageio

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson import object_states

def move_to_grasp_pose(grasp_action=1.0):
    # ======================= Move hand to grasp pose ================================    
    print("init_hand_pose: ", robot.get_relative_eef_pose(arm='right')[0])
    robot_to_box = np.array([
            [ 0.24697646,  0.96798984,  0.04470224, -0.02945701],
            [ 0.14370723,  0.00903325, -0.98957902,  0.00897374],
            [-0.95830625,  0.25082675, -0.13687614,  0.12712793],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ])
    box_pos, box_orn = box.get_position_orientation()
    box_to_world = np.eye(4)
    box_to_world[:3, :3] = R.from_quat(box_orn).as_matrix()
    box_to_world[:3, 3] = np.transpose(box_pos)
    robot_to_world = np.dot(box_to_world, robot_to_box)
    robot_to_world_orn = th.tensor(R.from_matrix(robot_to_world[:3, :3]).as_quat(), dtype=th.float32)
    robot_to_world_pos = th.tensor(robot_to_world[:3, 3], dtype=th.float32)
    target_pose = (robot_to_world_pos, robot_to_world_orn)
    
    pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
    execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                    env, 
                    robot, 
                    grasp_action, 
                    ) 
    
    execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                    env, 
                    robot, 
                    grasp_action, 
                    ) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return concatenated_image

def execute_controller(ctrl_gen, env, robot, grasp_action, episode_memory=None):
    obs, info = env.get_obs()
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        obs, reward, terminated, truncated, info = env.step(action)
        img = obs[f"{env.robots[0].name}"][f"{env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255
        viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255
        concat_img = hori_concatenate_image([viewer_img, img])
        writer.append_data(concat_img)

    return obs, info


def correct_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=100.0,
        dynamic_friction=100.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    # base_yaw = 90
    # r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    # r_quat = R.as_quat(r_euler)
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # # randomizing base pos
    # base_pos = np.array([-0.05, -0.4, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.503, -0.997])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


set_all_seeds(seed=3)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
# config["scene"] = dict()
# config["scene"]["type"] = "Scene"
# config["scene"]["scene_model"] = "Rs_int"
# config["scene"]["load_object_categories"] = ["floors", "ceilings", "coffee_table", "breakfast_table", "pot_plant", "laptop", "floor_lamp", "table_lamp"]
config["scene"]["load_object_categories"] = ["floors", "ceilings", "coffee_table", "breakfast_table"]

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_absolute_ori'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, 0.0]
box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
config["objects"] = [
    # {
    #     "type": "DatasetObject",
    #     "name": "shelf",
    #     "category": "shelf",
    #     "model": "eniafz",
    #     "position": [1.5, 0, 1.0],
    #     "scale": [2.0, 2.0, 1.0],
    #     "orientation": rot_quat,
    # },   
    # {
    #     "type": "DatasetObject",
    #     "name": "coffee_table",
    #     "category": "coffee_table",
    #     "model": "fqluyq",
    #     # "scale": [0.3, 0.3, 0.3],
    #     "position": [0, 0.6, 0.3],
    #     "orientation": [0, 0, 0, 1]
    # },
    # {
    #     "type": "PrimitiveObject",
    #     "name": "box",
    #     "primitive_type": "Cube",
    #     "rgba": [1.0, 0, 0, 1.0],
    #     "scale": [0.1, 0.05, 0.1],
    #     "position": [0.1, 0.5, 0.5],
    #     "orientation": box_quat
    # },
    {
        "type": "DatasetObject",
        "name": "box_of_baking_powder",
        "category": "box_of_baking_powder",
        "model": "vzgrlv",
        "position": [0.1, 0.5, 0.5],
        "scale": [0.5, 0.5, 1.0],
        "orientation": box_quat
    },
    {
        "type": "DatasetObject",
        "name": "plate",
        "category": "plate",
        "model": "ujodgo",
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

# og.sim.restore(["moma_pick_and_place/temp.json"])
# og.sim.restore(["temp2.json"])
# og.sim.restore(["nav_test_temp.json"])


coffee_table = env.scene.object_registry("name", "coffee_table_fqluyq_0")
breakfast_table = env.scene.object_registry("name", "breakfast_table_skczfi_0")

# fridge = env.scene.object_registry("name", "fridge_xyejdx_0")
# laptop = env.scene.object_registry("name", "laptop_nvulcs_0")
# table_lamp = env.scene.object_registry("name", "table_lamp_xbfgjc_0")
box = env.scene.object_registry("name", "box_of_baking_powder")
plate = env.scene.object_registry("name", "plate")
# pot_plant = env.scene.object_registry("name", "pot_plant_jatssq_0")
# pot_plant2 = env.scene.object_registry("name", "pot_plant_jatssq_1")
# floor_lamp = env.scene.object_registry("name", "floor_lamp_vdxlda_0")

box.states[object_states.OnTop].set_value(coffee_table, True)
plate.states[object_states.OnTop].set_value(coffee_table, True)

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([1.34,  -2.5,  1.41]),
    th.tensor([0.52,  0.24, 0.34, 0.73]),
)

scene = env.scene
robot = env.robots[0]
correct_gripper_friction()
# shelf = env.scene.object_registry("name", "shelf")
# shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

init_pose = robot.get_relative_eef_pose(arm='right')

# for saving videos
current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
base_folder = f"{current_date}"
time_folder = os.path.join(base_folder, current_time)
folder_path = f"outputs_data_gen/{time_folder}"
os.makedirs(folder_path, exist_ok=True)

imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
output_path = f'{folder_path}/video.mp4'
writer = imageio.get_writer(output_path, **imgio_kargs)


for _ in range(300):
    # for name, ctrl in robot._controllers.items():
    #     print(name, robot._controllers[name]._goal)
    # print(robot._controllers["arm_right"]._goal)
    og.sim.step()

post_eef_pose = robot.get_relative_eef_pose(arm='right')
pos_error = np.linalg.norm(post_eef_pose[0] - init_pose[0])
orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], init_pose[1])
print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

grasp_action = -1
init_pose = robot.get_relative_eef_pose(arm='right')

# # If want to run without the object
# robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
# for _ in range(20):
#     og.sim.step()

state = og.sim.dump_state()

# objs = [pot_plant2, laptop, box, pot_plant, floor_lamp]
objs = [box]
grasp_action = 1.0
for i, obj in enumerate(objs):
    print(f"Navigating to {obj.name}")
    # input()
    execute_controller(action_primitives._navigate_to_obj(obj),
                        env, robot, grasp_action=grasp_action)
    
    if i == 0:
        move_to_grasp_pose()
        # ============= Perform grasp ===================
        grasp_action = -1.0
        action = action_primitives._empty_action()
        action[robot.gripper_action_idx["right"]] = grasp_action
        env.step(action)
        for _ in range(40):
            og.sim.step()
        grasp_action = -1.0

        # ======================= Move hand up ================================  
        curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
        new_pos = curr_pos + th.tensor([0.0, 0.0, 0.2])
        target_pose = (new_pos, curr_orn)
        execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
                        env, 
                        robot, 
                        grasp_action, 
                        )
        
        for _ in range(40):
            og.sim.step()

        
        # og.sim.save([f'nav_test_temp.json'])

# # ======================= Move hand up ================================  
# curr_pos, curr_orn = robot.get_relative_eef_pose(arm='right')
# new_pos = curr_pos + th.tensor([0.0, 0.0, 0.2])
# target_pose = (new_pos, curr_orn)
# execute_controller(action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False), 
#                 env, 
#                 robot, 
#                 grasp_action, 
#                 )
# for _ in range(40):
#     og.sim.step()

print("place insideeeeee")
execute_controller(action_primitives._place_inside(plate), 
                        env, 
                        robot, 
                        grasp_action=-1.0, 
                        )
    
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
    #     # print("action: ", action)
    #     env.step(action=action)
    #     if keypress_str == 'TAB':
    #         right_eef_pose = robot.get_relative_eef_pose(arm='right')
    #         right_eef_pose_world = robot.eef_links["right"].get_position_orientation()
    #         base_pose = robot.get_position_orientation()
    #         print("right_eef_pose: ", right_eef_pose)
    #         print("right_eef_pose_world: ", right_eef_pose_world)
    #         print("base_pose: ", base_pose)
    #         box_pos, box_orn = box.get_position_orientation()
    #         robot_to_world = np.eye(4)
    #         robot_to_world[:3, :3] = R.from_quat(right_eef_pose_world[1]).as_matrix()
    #         robot_to_world[:3, 3] = np.transpose(right_eef_pose_world[0])
    #         box_to_world = np.eye(4)
    #         box_to_world[:3, :3] = R.from_quat(box_orn).as_matrix()
    #         box_to_world[:3, 3] = np.transpose(box_pos)
    #         robot_to_box = np.dot(np.linalg.inv(box_to_world), robot_to_world)
    #         print("robot_to_box: ", robot_to_box)
    #     step += 1
    # # ========================================================================
    

for i in range(200):
    og.sim.step()


og.shutdown()
