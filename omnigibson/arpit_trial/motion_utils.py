import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from memory import Memory
from utils import dump_to_memory

class MotionUtils:
    def __init__(self, env, robot, action_primitives, writer):
        self.env = env
        self.robot = robot
        self.action_primitives = action_primitives
        self.writer = writer

    def undo_action(self, t, action):
        print("Undoing action")
        a = th.cat((-action[t][:-1], action[t][-1:])) 
        self.move_primitive(a, ik_test=False)

    def execute_controller(self, ctrl_gen, grasp_action, episode_memory=None):
        obs, info = self.env.get_obs()          
        total_collisions = 0
        singularities = []
        reached_singularity = False
        for action in ctrl_gen:
            if action == 'Done':
                if episode_memory is not None:
                    dump_to_memory(self.env, self.robot, episode_memory) 
                continue
            action[self.robot.gripper_action_idx["right"]] = grasp_action
            # print("action: ", action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            img = obs[f"{self.env.robots[0].name}"][f"{self.env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
            self.writer.append_data(img)
            box = self.env.scene.object_registry("name", "box")
            arm_in_collision = detect_robot_collision_in_sim(self.robot, filter_objs=[box])
            if arm_in_collision:
                total_collisions += 1
            
            singularity = self.robot._controllers["arm_right"].singularity
            singularities.append(singularity)

            if sum(singularities) > 3:
                reached_singularity = True
                return obs, info, total_collisions, reached_singularity
            
            # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
            # print("normalized_qpos: ", normalized_qpos)
        return obs, info, total_collisions, reached_singularity

    def move_primitive(self, action, episode_memory=None, ik_test=True):
        incorrect_control = False

        current_pose = self.robot.get_relative_eef_pose(arm='right')
        current_pos = current_pose[0]
        current_orn = current_pose[1]
        
        delta_pos = action[3:6]
        delta_orn = action[6:9]
        grasp_action = action[9]
        
        target_pos = current_pos + delta_pos
        target_pos = target_pos.type(th.FloatTensor)
        target_orn = R.from_quat(R.from_rotvec(delta_orn).as_quat()) * R.from_quat(current_orn)
        target_orn = th.tensor(target_orn.as_quat())
        target_orn = target_orn.type(th.FloatTensor)

        target_pose = (target_pos, target_orn)

        # test_joint_pos = action_primitives._ik_solver_cartesian_to_joint_space(target_pose)
        # print("test_joint_pos: ", test_joint_pos)
        # if ik_test and test_joint_pos is None:
        #     safe = False
        #     return None, None, 0, safe
        
        obs, info, total_collisions1, reached_singularity1 = self.execute_controller(self.action_primitives._move_hand_direct_ik(target_pose,
                                                                                stop_on_contact=False,
                                                                                ignore_failure=True,
                                                                                stop_if_stuck=False,
                                                                                in_world_frame=False), 
                                                                        grasp_action, 
                                                                        episode_memory)  

        # obtain target pose2d
        current_base_pos, current_base_orn_quat = self.robot.get_position_orientation()
        current_base_yaw = R.from_quat(current_base_orn_quat).as_euler('XYZ')[2]

        delta_base_pos = action[0:2] # this is in the robot frame
        # conver delta pos from robot frame to world frame
        robot_to_world = np.eye(4)
        robot_to_world[:3, :3] = R.from_quat(current_base_orn_quat).as_matrix()
        robot_to_world[:3, 3] = np.transpose(np.array([0.0, 0.0, 0.0]))
        delta_base_pos_homo = np.array([delta_base_pos[0], delta_base_pos[1], 0.0, 1.0])
        delta_base_pos_world = np.dot(robot_to_world, delta_base_pos_homo)
        delta_base_pos_world = th.from_numpy(delta_base_pos_world)
        delta_base_yaw = action[2]

        target_base_pos = current_base_pos + th.tensor([delta_base_pos[0], delta_base_pos[1], 0.0])
        target_base_yaw = current_base_yaw + delta_base_yaw
        target_pose2d = th.tensor([target_base_pos[0], target_base_pos[1], target_base_yaw])
        obs, info, total_collisions2, reached_singularity2 = self.execute_controller(self.action_primitives._navigate_to_pose_direct(target_pose2d), 
                        grasp_action, 
                        episode_memory)


        # Hack to ensure that even if primitive does not return any action (if delta pose is 0), grasp action is performed
        action = self.action_primitives._empty_action()
        obs, info, total_collisions3, reached_singularity3 = self.execute_controller([action], 
                        grasp_action, 
                        episode_memory)

        total_collisions = max(total_collisions1, total_collisions2, total_collisions3)
        reached_singularity = reached_singularity1 or reached_singularity2 or reached_singularity3
        print("total_collisions: ", total_collisions)

        for _ in range(50):
            og.sim.step()

        ee_pose_after = self.robot.get_relative_eef_pose(arm='right')
        pos_error = np.linalg.norm(ee_pose_after[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], target_pose[1])
        orn_error = orn_error % (2*th.pi)
        print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")

        if pos_error > 0.05 or orn_error > 0.2:
            incorrect_control = True

        return obs, info, total_collisions, incorrect_control, reached_singularity

    def first_primitive(self, primitive_steps_to_perform, episode_memory=None, grasp_mode="vertical"):

        grasp_action = 1.0
        if 1 in primitive_steps_to_perform:
            # ======================= Move base ================================  
            grasp_action = 1.0
            # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
            target_base_pose = th.tensor([0.0, 0.0, 1.57])
            self.execute_controller(self.action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory, grasp_action=grasp_action), 
                            grasp_action, 
                            episode_memory)    
            # for _ in range(50):
            #     og.sim.step()
            curr_base_pos = self.robot.get_position_orientation()[0]
            print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
            # =================================================================================

        grasp_sim_state = og.sim.dump_state()
        
        if 2 in primitive_steps_to_perform:
            self.move_to_grasp_pose(grasp_mode, grasp_action)

        if 3 in primitive_steps_to_perform:
            # ============= Perform grasp ===================
            grasp_action = -1.0
            action = self.action_primitives._empty_action()
            action[self.robot.gripper_action_idx["right"]] = grasp_action
            self.env.step(action)
            for _ in range(40):
                og.sim.step()
            if episode_memory is not None:
                # save everything to memory
                dump_to_memory(self.env, self.robot, episode_memory)
                # TODO: Change the indexing here
                action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[14:21]))) # TODO check the indices here    
                episode_memory.add_action('actions', action_to_add)
            # ==============================================
            
        if 4 in primitive_steps_to_perform:
            # ======================= Move hand up ================================  
            curr_pos, curr_orn = self.robot.get_relative_eef_pose(arm='right')
            new_pos = curr_pos + th.tensor([0.0, 0.0, 0.2])
            target_pose = (new_pos, curr_orn)
            self.execute_controller(self.action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=False, episode_memory=episode_memory), 
                            grasp_action, 
                            episode_memory)
            
            for _ in range(40):
                og.sim.step()
            
            # Debugging
            post_eef_pose = self.robot.get_relative_eef_pose(arm='right')
            pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
            orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
            print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
            # =================================================================================

        if 5 in primitive_steps_to_perform:
            # ============= Move base ===================
            # debugging
            ee_pose_before_nav = self.robot.get_relative_eef_pose(arm='right')
            # target_base_pose = (th.tensor([0.4256, 0.0257, 0.0005]), th.tensor([-6.8379e-08, -7.3217e-08,  3.1305e-02,  9.9951e-01]))
            target_base_pose = th.tensor([0.486, 0.0257, 0.0]) # [0.456, 0.0257, 0.0] [0.526, 0.0257, 0.0]
            self.execute_controller(self.action_primitives._navigate_to_pose_linearly_cartesian(target_base_pose, episode_memory=episode_memory, grasp_action=grasp_action),
                            grasp_action, 
                            episode_memory)    
            for _ in range(50):
                og.sim.step()
            curr_base_pos = self.robot.get_position_orientation()[0]
            print("move base completed. Final right eef pose reached: ", target_base_pose[:2], curr_base_pos[:2])
            
            # Debugging
            ee_pose_after_nav = self.robot.get_relative_eef_pose(arm='right')
            pos_error = np.linalg.norm(ee_pose_after_nav[0] - ee_pose_before_nav[0])
            orn_error = T.get_orientation_diff_in_radian(ee_pose_after_nav[1], ee_pose_before_nav[1])
            print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
            # ============================================

        return grasp_sim_state

    def perform_grasp(self, episode_memory=None):
        # ======================= Move hand to grasp pose ================================    
        grasp_action = 1.0
        # w.r.t world
        target_pose = (th.tensor([0.1829, 0.4876, 0.4051]), th.tensor([-0.0342, -0.0020,  0.9958,  0.0846]))
        # w.r.t robot
        # target_pose = (th.tensor([ 0.4976, -0.2129,  0.4346]), th.tensor([-0.0256,  0.0228,  0.6444,  0.7640]))
        # # diagonal 45
        # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
        self.execute_controller(self.action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                            grasp_action, 
                            episode_memory) 
        for _ in range(40):
            og.sim.step()
        # current_pose_world = robot.eef_links["right"].get_position_orientation()
        # print("move hand down completed. Desired and Reached right eef pose reached: ", target_pose[0], current_pose_world[0])
        # =================================================================================
            
    def move_to_grasp_pose(self, grasp_mode, grasp_action, episode_memory=None):
        # ======================= Move hand to grasp pose ================================    
        print("init_hand_pose: ", self.robot.get_relative_eef_pose(arm='right')[0])
        if grasp_mode == "horizontal":
            # horizontal
            # w.r.t robot
            # target_pose:  (th.tensor([ 0.4891, -0.1747,  0.3917]), th.tensor([-0.0224,  0.0234,  0.6525,  0.7571]))
            # w.r.t world
            target_pose = (th.tensor([0.1747, 0.4891, 0.3922]), th.tensor([-3.2351e-02,  6.7136e-04,  9.9674e-01,  7.3904e-02]))
        
        # # diagonal 45
        # target_pose = (th.tensor([0.1442, 0.4779, 0.4515]), th.tensor([-0.0614, -0.8765, -0.0655, -0.4730]))
        
        if grasp_mode == "vertical":
            # vertical
            # w.r.t robot
            # target_pose = (th.tensor([ 0.5066, -0.0575,  0.4948]), th.tensor([ 0.4775,  0.5259, -0.5041,  0.4913]))
            # w.r.t world
            target_pose = (th.tensor([0.0933, 0.5011, 0.4953]), th.tensor([ 0.0090, -0.7102,  0.0341, -0.7031]))

        pre_target_pose = (target_pose[0] + th.tensor([0.0, 0.0, 0.1]), target_pose[1]) 
        self.execute_controller(self.action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
                        grasp_action, 
                        episode_memory) 
        
        self.execute_controller(self.action_primitives._move_hand_direct_ik(target_pose, ignore_failure=True, in_world_frame=True), 
                        grasp_action, 
                        episode_memory) 
        for _ in range(40):
            og.sim.step()
        
        # Debugging
        # post_eef_pose = robot.get_relative_eef_pose(arm='right')
        post_eef_pose = self.robot.eef_links["right"].get_position_orientation()
        pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
        orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
        print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
        # =================================================================================

    def custom_reset(self, env, robot, episode_memory=None): 
        # scene_initial_state = env.scene._initial_state
        
        # base_yaw = 90
        # r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
        # r_quat = R.as_quat(r_euler)
        # scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat
        
        # # Reset environment and robot
        # env.reset()
        # robot.reset()

        # set head joint positions
        head_joints = th.tensor([-0.503, -0.857]) #-0.503, -0.897
        robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

        # Step simulator a few times so that the effects of "reset" take place
        for _ in range(10):
            og.sim.step()

    def safe(self, action, use_hack=False, collision_failure_model=None, grasp_mode=None):
        safe = True
        unsafe_reasons = []
        prev_state = og.sim.dump_state()
        box = self.env.scene.object_registry("name", "box")
        obj_in_hand_pos_before = box.get_position_orientation()[0]

        # Using model to check for collisions ------
        obs, obs_info = self.env.get_obs()
        if collision_failure_model is not None:
            if grasp_mode == "vertical":
                threshold = 0.0
            elif grasp_mode == "horizontal":
                threshold = 0.6
            check_collision = collision_failure_model.check_collision(obs, obs_info, action, self.env.robots[0].name, threshold=threshold)
            if check_collision == 1.0:
                safe = False
                unsafe_reasons.append("Model says will collide") 
                return safe
        # --------------------------------
        
        
        _, _, total_collisions, incorrect_control, reached_singularity = self.move_primitive(action)

        if use_hack:
            self.robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=self.robot.gripper_control_idx['right'])
            action = self.action_primitives._empty_action()
            action[self.robot.gripper_action_idx["right"]] = 1.0
            self.env.step(action)
            for _ in range(40):
                # print(robot._controllers["arm_right"]._goal)
                og.sim.step()
                obs, _ = self.env.get_obs()
                img = obs[f"{self.env.robots[0].name}"][f"{self.env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy()
                self.writer.append_data(img)

            
            # gripper_pos = robot.get_joint_positions()[robot.gripper_control_idx["right"]]
            # print("gripper finger joint positions after opening: ", gripper_pos)
            # if abs(gripper_pos[0] - 0.045) > 0.01 or abs(gripper_pos[1] - 0.045) > 0.01:
                # input("GRIPPER DID NOT OPEN!!. Press enter to continue")
        
        obj_in_hand_pos_after = box.get_position_orientation()[0]
        delta_pos_z = abs(obj_in_hand_pos_before[2] - obj_in_hand_pos_after[2]) 
        # print("delta_pos_z: ", delta_pos_z)
        # print("total_collisions: ", total_collisions)

        normalized_qpos = self.robot.get_joint_positions(normalized=True)[self.robot.arm_control_idx["right"]]
        # print("normalized_qpos: ", normalized_qpos)
        close_to_one = th.isclose(normalized_qpos[:-3], th.tensor(1.0), atol=1e-2)
        close_to_neg_one = th.isclose(normalized_qpos[:-3], th.tensor(-1.0), atol=1e-2)
        any_close_to_one_or_neg_one = (close_to_one | close_to_neg_one).any().item()
        if any_close_to_one_or_neg_one:
            safe = False
            unsafe_reasons.append("Reaching some joint limit") 

        # object dropped (unsafe)
        if delta_pos_z > 0.35:
            safe = False 
            unsafe_reasons.append("Will drop object") 

        # # collisions
        # if total_collisions > 0:
        #     safe = False
        #     unsafe_reasons.append("Will collide") 
        #     print("In reality total_collisions: ", total_collisions)
        # else:
        #     print("In reality, no collisions")
        # breakpoint()

        # # replace collision checking with a learned model
        # obs, obs_info = self.env.get_obs()
        # if collision_failure_model is not None:
        #     check_collision = collision_failure_model.check_collision(obs, obs_info, action, self.env.robots[0].name)
        #     if check_collision:
        #         safe = False
        #         unsafe_reasons.append("Will collide") 
        
        # singularities. Need to do this as using IK solver to test if a target pose is reachable is not working well.
        if reached_singularity:
            safe = False
            unsafe_reasons.append("Will reach singularity") 

        if incorrect_control:
            safe = False
            unsafe_reasons.append("Will lead to incorrect control (Let's skip this action).") 


        print("is this action safe? ", safe, unsafe_reasons)
        if not safe:
            # Hack to make sure that load_state will work. I think there is an issue in using og.sim.load_state() when there are weird collisions
            self.robot.set_position_orientation(position=th.tensor([-2.0, 0.0, 0.0]))
            og.sim.load_state(prev_state)
            for _ in range(30):
                og.sim.step()
            print("Reset state via og.sim.load_state()")
            # breakpoint()
        
        # input()
        return safe