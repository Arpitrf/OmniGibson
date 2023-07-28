import numpy as np
from ompl import base as ob
from ompl import geometric as ompl_geo

import omnigibson as og
from omnigibson.object_states import ContactBodies
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import RigidContactAPI
from pxr import PhysicsSchemaTools, Gf

import time

PLANNERS = {
    "RRTConnect" : ompl_geo.RRTConnect, 
    "RRTstar" : ompl_geo.RRTstar, # optim
    "RRTsharp" : ompl_geo.RRTsharp, # optim: faster-convergence version of RRT*
    "RRTXstatic" : ompl_geo.RRTXstatic, # optim: faster-convergence version of RRT*
    "BITstar": ompl_geo.BITstar, # optim
}

OPTIM_PLANNERS = [
    "RRTstar", "RRTsharp", "RRTXstatic",
    "BITstar",
]

# class WeightedJointMotionOptimizationObjective(ob.PathLengthOptimizationObjective):
#     def __init__(self, weights):
#         self.weights = weights

#     def motionCost(self, s1, s2):
#         cost = 0
#         for i in len(s1 - 1):
#             cost += abs(s2[i] - s1[i]) * self.weights[i]
#         return ob.Cost(cost)


def plan_base_motion(
    robot,
    end_conf,
    context,
    planning_time = 100.0,
    **kwargs
):
    def state_valid_fn(q):
        x = q.getX()
        y = q.getY()
        yaw = q.getYaw()
        pose = ([x, y, 0.0], T.euler2quat((0, 0, yaw)))
        state_valid = not detect_robot_collision(context, pose)
        return state_valid

    pos = robot.get_position()
    yaw = T.quat2euler(robot.get_orientation())[2]
    start_conf = (pos[0], pos[1], yaw)

    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-3.0)
    bounds.setHigh(3.0)
    space.setBounds(bounds)

    # create a simple setup object
    ss = ompl_geo.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

    si = ss.getSpaceInformation()
    planner = ompl_geo.RRTConnect(si)
    ss.setPlanner(planner)

    start = ob.State(space)
    start().setX(start_conf[0])
    start().setY(start_conf[1])
    start().setYaw(T.wrap_angle(start_conf[2]))
    print(start)

    goal = ob.State(space)
    goal().setX(end_conf[0])
    goal().setY(end_conf[1])
    goal().setYaw(T.wrap_angle(end_conf[2]))
    print(goal)

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(planning_time)

    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        sol_path = ss.getSolutionPath()
        return_path = []
        for i in range(sol_path.getStateCount()):
            x = sol_path.getState(i).getX()
            y = sol_path.getState(i).getY()
            yaw = sol_path.getState(i).getYaw()
            return_path.append([x, y, yaw])
        return remove_unnecessary_rotations(return_path)
    return None


def plan_arm_motion(
    robot,
    end_conf,
    context,
    planning_time = 20.0,
    algo="BITstar",
    simplifiers=[],
    setrange=0.0,
    **kwargs,
):
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    dim = len(joint_control_idx)

    if "combined" in robot.robot_arm_descriptor_yamls:
        joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
        control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
    else:
        initial_joint_pos = np.array(robot.get_joint_positions()[joint_control_idx])
        control_idx_in_joint_pos = np.arange(dim)

    # state validity function (collision checker)
    def state_valid_fn(q):
        joint_pos = initial_joint_pos
        joint_pos[control_idx_in_joint_pos] = [q[i] for i in range(dim)]
        return arm_planning_validity_fn(context, joint_pos)
    
    # breakpoint()

    # create an SE2 state space
    space = ob.RealVectorStateSpace(dim)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(dim)
    joints = np.array([joint for joint in robot.joints.values()])
    arm_joints = joints[joint_control_idx]
    for i, joint in enumerate(arm_joints):
        if end_conf[i] > joint.upper_limit:
            end_conf[i] = joint.upper_limit
        if end_conf[i] < joint.lower_limit:
            end_conf[i] = joint.lower_limit
        bounds.setLow(i, float(joint.lower_limit))
        bounds.setHigh(i, float(joint.upper_limit))
    space.setBounds(bounds)

    # get state information
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))
    si.setup()

    # create problem definition 
    pdef = ob.ProblemDefinition(si)

    # set start and goal states
    start_conf = robot.get_joint_positions()[joint_control_idx]
    start = ob.State(space)
    for i in range(dim):
        start[i] = float(start_conf[i])

    goal = ob.State(space)
    for i in range(dim):
        goal[i] = float(end_conf[i])
    pdef.setStartAndGoalStates(start, goal)

    # setup optimization objective
    optim_obj = ob.PathLengthOptimizationObjective(si)
    pdef.setOptimizationObjective(optim_obj)

    # define optimizing planner
    planner = PLANNERS[algo](si)
    planner.setProblemDefinition(pdef)
    try:
        planner.setRange(setrange)
    except:
        print("this planner does not have setRange")
    planner.setup()

    # define path simplifier
    ps = ompl_geo.PathSimplifier(si)
    SIMPLIFIERS = {
        "reduceVertices": ps.reduceVertices,
        "shortcutPath": ps.shortcutPath,
        "collapseCloseVertices": ps.collapseCloseVertices,
        "smoothBSpline": ps.smoothBSpline,
        "simplifyMax": ps.simplifyMax,
        "findBetterGoal": ps.findBetterGoal,
    }

    solved = planner.solve(planning_time)

    if solved:
        # try to shorten the path
        simp_start_time = time.time()
        for simplifier in simplifiers:
            cur_time = time.time()
            print("----------Running simplifier ", simplifier, "----------")
            SIMPLIFIERS[simplifier](pdef.getSolutionPath())
            print("Simplifier ", simplifier, " took ", time.time() - cur_time, " seconds")
        print("All simplifications combined took ", time.time() - simp_start_time, " seconds")
        sol_path = pdef.getSolutionPath()

        return_path = []
        for i in range(sol_path.getStateCount()):
            joint_pos = [sol_path.getState(i)[j] for j in range(dim)]
            return_path.append(joint_pos)
        return return_path
    return None




# def plan_arm_motion(
#     robot,
#     end_conf,
#     context,
#     planning_time = 20.0,
#     algo="BITstar",
#     simplifiers=[],
#     setrange=0.0,
#     **kwargs,
# ):
#     joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
#     dim = len(joint_control_idx)

#     if "combined" in robot.robot_arm_descriptor_yamls:
#         joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
#         initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
#         control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
#     else:
#         initial_joint_pos = np.array(robot.get_joint_positions()[joint_control_idx])
#         control_idx_in_joint_pos = np.arange(dim)

#     def state_valid_fn(q):
#         joint_pos = initial_joint_pos
#         joint_pos[control_idx_in_joint_pos] = [q[i] for i in range(dim)]
#         return arm_planning_validity_fn(context, joint_pos)
    
#     # create an SE2 state space
#     space = ob.RealVectorStateSpace(dim)

#     # set lower and upper bounds
#     bounds = ob.RealVectorBounds(dim)
#     joints = np.array([joint for joint in robot.joints.values()])
#     arm_joints = joints[joint_control_idx]
#     for i, joint in enumerate(arm_joints):
#         if end_conf[i] > joint.upper_limit:
#             end_conf[i] = joint.upper_limit
#         if end_conf[i] < joint.lower_limit:
#             end_conf[i] = joint.lower_limit
#         bounds.setLow(i, float(joint.lower_limit))
#         bounds.setHigh(i, float(joint.upper_limit))
#     space.setBounds(bounds)

#     # create a simple setup object
#     ss = ompl_geo.SimpleSetup(space)
#     ss.setStateValidityChecker(ob.StateValidityCheckerFn(state_valid_fn))

#     # get state information
#     si = ss.getSpaceInformation()

#     # define planner
#     planner = PLANNERS[algo](si)

#     # planner.setRange(0.01)
#     ss.setPlanner(planner)
    
#     start_conf = robot.get_joint_positions()[joint_control_idx]
#     start = ob.State(space)
#     for i in range(dim):
#         start[i] = float(start_conf[i])

#     goal = ob.State(space)
#     for i in range(dim):
#         goal[i] = float(end_conf[i])
#     ss.setStartAndGoalStates(start, goal)

#     # # setup optimization objective
#     # if algo in OPTIM_PLANNERS:
#     #     optim_obj = ob.PathLengthOptimizationObjective(si)
#     #     ss.setOptimizationObjective(optim_obj)
#     #     planner.setProblemDefinition(si)
#     #     planner.setup()

#     # define path simplifier
#     ps = ompl_geo.PathSimplifier(si)
#     SIMPLIFIERS = {
#         "reduceVertices": ps.reduceVertices,
#         "shortcutPath": ps.shortcutPath,
#         "collapseCloseVertices": ps.collapseCloseVertices,
#         "smoothBSpline": ps.smoothBSpline,
#         "simplifyMax": ps.simplifyMax,
#         "findBetterGoal": ps.findBetterGoal,
#     }

#     solved = ss.solve(planning_time)

#     if solved:
#         # try to shorten the path
#         # ss.simplifySolution()
#         simp_start_time = time.time()
#         for simplifier in simplifiers:
#             cur_time = time.time()
#             print("----------Running simplifier ", simplifier, "----------")
#             SIMPLIFIERS[simplifier](ss.getSolutionPath())
#             print("Simplifier ", simplifier, " took ", time.time() - cur_time, " seconds")
#         print("All simplifications combined took ", time.time() - simp_start_time, " seconds")
        
#         sol_path = ss.getSolutionPath()

#         return_path = []
#         for i in range(sol_path.getStateCount()):
#             joint_pos = [sol_path.getState(i)[j] for j in range(dim)]
#             return_path.append(joint_pos)
#         return return_path
#     return None

# Moves robot and detects robot collisions with the environment, but not with itself
def detect_robot_collision(context, pose):
    translation = pose[0]
    orientation = pose[1]
    # context.robot_copy.prim.set_local_poses(np.array([translation]), np.array([orientation]))
    translation = Gf.Vec3d(*np.array(translation, dtype=float))
    context.robot_copy.prim.GetAttribute("xformOp:translate").Set(translation)

    orientation = np.array(orientation, dtype=float)[[3, 0, 1, 2]]
    context.robot_copy.prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation)) 
                
    for link in context.robot_meshes_copy:
        for mesh in context.robot_meshes_copy[link]:
            mesh_id = PhysicsSchemaTools.encodeSdfPath(mesh.prim_path)
            if mesh._prim.GetTypeName() == "Mesh":
                if og.sim.psqi.overlap_mesh_any(*mesh_id):
                    return True
            else:
                if og.sim.psqi.overlap_shape_any(*mesh_id):
                    return True
    return False

# Detects robot collisions with the environment, but not with itself
def detect_robot_collision_in_sim(robot, filter_objs=[]):
    filter_categories = ["floors"]
    
    obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    if obj_in_hand is not None:
        filter_objs.append(obj_in_hand)

    collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

    for col_prim in collision_prims:
        tokens = col_prim.prim_path.split("/")
        obj_prim_path = "/".join(tokens[:-1])
        col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
        if col_obj.category in filter_categories:
            collision_prims.remove(col_prim)

    return len(collision_prims) > 0
    

# Sets joint positions of robot and detects robot collisions with the environment and itself
def arm_planning_validity_fn(context, joint_pos):
    arm_links = context.robot.manipulation_link_names
    link_poses = context.fk_solver.get_link_poses(joint_pos, arm_links)

    for link in arm_links:
        pose = link_poses[link]
        if link in context.robot_meshes_copy.keys():
            for mesh, relative_pose in zip(context.robot_meshes_copy[link], context.robot_meshes_relative_poses[link]):
                mesh_pose = T.pose_transform(*pose, *relative_pose)
                translation = Gf.Vec3d(*np.array(mesh_pose[0], dtype=float))
                mesh._prim.GetAttribute("xformOp:translate").Set(translation)
                orientation = np.array(mesh_pose[1], dtype=float)[[3, 0, 1, 2]]
                mesh._prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation))

    # Define function for checking overlap
    valid_hit = False
    mesh_hit = None

    def overlap_callback(hit):
        nonlocal valid_hit
        nonlocal mesh_hit
        
        valid_hit = hit.rigid_body not in context.disabled_collision_pairs_dict[mesh_hit]

        return not valid_hit

    def check_overlap():
        nonlocal valid_hit
        nonlocal mesh_hit
        valid_hit = False

        for link in context.robot_meshes_copy:
            for mesh in context.robot_meshes_copy[link]:
                if valid_hit:
                    return valid_hit
                mesh_id = PhysicsSchemaTools.encodeSdfPath(mesh.prim_path)
                mesh_hit = mesh.prim_path
                if mesh._prim.GetTypeName() == "Mesh":
                    og.sim.psqi.overlap_mesh(*mesh_id, reportFn=overlap_callback)
                else:
                    og.sim.psqi.overlap_shape(*mesh_id, reportFn=overlap_callback)
            
        return valid_hit
    
    return not check_overlap()

def remove_unnecessary_rotations(path):
    for start_idx in range(len(path) - 1):
        start = np.array(path[start_idx][:2])
        end = np.array(path[start_idx + 1][:2])
        segment = end - start
        theta = np.arctan2(segment[1], segment[0])
        path[start_idx] = (start[0], start[1], theta)
    return path