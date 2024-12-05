import math
import numpy as np

from scipy.spatial.transform import Rotation as R

video_name = "nav_test8"
# "/home/arpit/projects/og_prior_npz_files/{data_seq}_prior_results.npz"
data = np.load(f"/home/arpit/projects/og_prior_npz_files/{video_name}_prior_results.npz")
body_trans = data["body_positions"]
body_orient = data["body_orientations"]
hand_positions = data["hand_positions"]
hand_rotations = data["hand_orientations"]

print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
print(f"hand_rotations shape: {hand_rotations.shape}")
print(f"body_trans shape: {body_trans.shape}")
print(f"body_orient shape: {body_orient.shape}\n\n\n")

# --------------------------------------- WORKING -------------------------------------------
points = [(t[0], t[2]) for t in body_trans]
root_orient = data["body_orientations"]

# Getting orientations
yaw = []
for i in range(root_orient.shape[0]):
    rotmat = R.from_euler("xyz", root_orient[i]).as_matrix()
    unit_vector = np.array([1., 0., 0.])
    direction_vector = np.matmul(rotmat, unit_vector)
    orientation = math.atan2(direction_vector[2], direction_vector[0])
    # yaw.append(orientation)
    yaw.append(orientation - 3.14/2)

init_matrix = np.array([
    [np.cos(yaw[0]), np.sin(yaw[0])],
    [-np.sin(yaw[0]), np.cos(yaw[0])]
])

points_init_frame = []
yaw_init_frame = []

for i in range(len(points)):
    delta_pos = np.array([points[i][0] - points[0][0], points[i][1] - points[0][1]])
    pos_robot_frame = np.matmul(init_matrix, np.transpose(delta_pos))
    # print("pos_robot_frame: ", pos_robot_frame)

    points_init_frame.append((pos_robot_frame[0], pos_robot_frame[1]))
    yaw_init_frame.append(yaw[i] - yaw[0])
    print("yaw[i] - yaw[0]: ", yaw[i] - yaw[0])
    # print("points_init_frame: ", points_init_frame)
# print("yaw_init_frame: ", yaw_init_frame)
# -------------------------------------------


# print("body_trans: ", body_trans)

# # test -----------
# T_base_init_to_camera = np.eye(4)
# # rotmat = R.from_euler("xyz", body_orient[0]).as_matrix()
# rotmat = R.from_rotvec(body_orient[0]).as_matrix()
# T_base_init_to_camera[:3, :3] = rotmat
# T_base_init_to_camera[:3, 3] = body_trans[0]

# T_base_to_camera_list = []
# T_base_to_base_init_list = []
# for i in range(1, body_orient.shape[0]):
#     T_base_to_camera = np.eye(4)
#     # rotmat = R.from_euler("xyz", body_orient[i]).as_matrix()
#     rotmat = R.from_rotvec(body_orient[i]).as_matrix()
#     T_base_to_camera[:3, :3] = rotmat
#     T_base_to_camera[:3, 3] = body_trans[i]
#     # T_base_to_camera_list.append(T_base_to_camera)  

#     T_base_to_base_init = np.linalg.inv(T_base_init_to_camera) @ T_base_to_camera
#     print("base pos: ", T_base_to_base_init[:2, 3])

# # ----------------


# test 2------------
# T_base_init_to_camera = np.eye(3)

# ----------------


# # TRANSFORMATION BY INIT POSE METHOD!!
# body_points = [(t[0], t[2]) for t in body_trans]
# body_yaw = []
# for i in range(body_orient.shape[0]):
#     rotmat = R.from_euler("xyz", body_orient[i]).as_matrix()
#     unit_vector = np.array([1., 0., 0.])
#     direction_vector = np.matmul(rotmat, unit_vector)
#     orientation = math.atan2(direction_vector[2], direction_vector[0])
#     # print("orientation: ", orientation)
#     body_yaw.append(orientation)
#     # body_yaw.append(orientation - 3.14/2)

# print("body_yaw[0]: ", body_yaw[0])
# init_matrix = np.array([
#     [np.cos(body_yaw[0]), -np.sin(body_yaw[0])],
#     [np.sin(body_yaw[0]), np.cos(body_yaw[0])]
# ])

# points_init_frame = []
# yaw_init_frame = []

# for i in range(len(body_points)):
#     delta_pos = np.array([body_points[i][0] - body_points[0][0], body_points[i][1] - body_points[0][1]])
#     # print("delta_pos: ", delta_pos)
#     pos_robot_frame = np.matmul(init_matrix, np.transpose(delta_pos))

#     points_init_frame.append((pos_robot_frame[0], pos_robot_frame[1]))
#     yaw_init_frame.append(body_yaw[i] - body_yaw[0])

# # breakpoint()

# frame_step = 10
# for i in range (frame_step, hand_positions.shape[0], frame_step):
#     current_pos, current_yaw = points_init_frame[i], yaw_init_frame[i]
#     prev_pos, prev_yaw = points_init_frame[i - frame_step], yaw_init_frame[i - frame_step]

#     delta_body_pos = np.array([current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]])
#     delta_yaw = current_yaw - prev_yaw

#     print(f"delta_body_pos: {delta_body_pos}")
#     # print(f"delta_yaw: {delta_yaw}")


# -----------------------------------

