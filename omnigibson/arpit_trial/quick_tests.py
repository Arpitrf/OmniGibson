import h5py
import numpy as np
import pickle

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)

f = h5py.File('moma_pick_and_place/dataset.hdf5', "r")
# ======================== Basic hdf5 testing ======================
print("len: ", len(f['data']))
# print("--", f['data/episode_00002/observations'].keys())
# print("--", np.array(f['data/episode_00002/observations/seg_semantic']).shape)
print(np.array(f['data/episode_00000/actions/actions']))
# print(np.array(f['data/episode_00003/extras/contacts']))
# print("--", np.array(f[f'data/episode_00000/observations_info']['seg_semantic']))
# print("joint efforts: ", np.array(f[f'data/episode_00000/proprioceptions/joint_qeffort']).shape)
# print("--", np.array(f[f'data/episode_00001/observations_info']['seg_instance_id_strings']))
# for ep in f['data'].keys():
#     print(ep, np.array(f['data'][f'{ep}/actions/actions']).shape, np.array(f['data'][f'{ep}/observations/rgb']).shape, np.array(f['data'][f'{ep}/extras/contacts']).shape)
#     print("ep: ", ep)
    # print("--", f[f'data/{ep}/observations_info'].keys())
# obs_info = f['data/episode_00000/observations_info'].keys()
# print("obs_info: ", obs_info)
# grasped_state = np.array(f['data/episode_00000/extras/grasps'])
# print("grasped_state: ", grasped_state)

# imgs = np.array(f[f'data/episode_00000/observations/rgb'])
# fig, ax = plt.subplots(1, imgs.shape[0])
# for i in range(len(imgs)):
#     ax[i].imshow(imgs[i])
# plt.show()
# ===================================================================

# # ======================= Episodes with collisions ======================
# num_contacts = 0
# num_no_contacts = 0
# for k in f['data'].keys():
#     contacts = np.array(f['data'][k]['extras']['contacts'])
#     for c in contacts:
#         if c:
#             num_contacts += 1
#         else:
#             num_no_contacts += 1
#     # print("grasp.shape: ", contacts.shape)
#     # print("k, contacts: ", k, contacts)
#     # print("----------------")
#     # if any(contacts):
# #     if contacts[-1]:
# #         print("k, contacts: ", k, contacts)
# #         succ_episodes += 1
# print("contacts and no_contacts: ", num_contacts, num_no_contacts)
# # ======================================================================


# # ==================== cem tests ====================
# num_top_samples = 3
# with open('outputs/run_2024-09-22_15-52-22/epoch_00.pkl', 'rb') as f:
#     x = pickle.load(f)
#     print(x.keys())

#     CEM_rewards = x['CEM_rewards'] 
#     print(x['CEM_rewards'])
#     print(x['mu_x'])
#     print(x['sigma_x'])

#     sorted_prediction_inds = np.argsort(-CEM_rewards.flatten())
#     top_indices = sorted_prediction_inds[:num_top_samples]
#     print("top_indices: ", top_indices)
#     print("top rewards: ", CEM_rewards[top_indices])
#     CEM_yaw_noises = x['CEM_yaw_noises']
#     print("CEM_yaw_noises: ", CEM_yaw_noises[top_indices])

#     print("mu and sigma for yaw: ", x['mu_yaw'], x['sigma_yaw'])
# # ===================================================
    

# # ============= checking point cloud ====================
# import open3d as o3d

# def generate_point_cloud_from_depth(depth_image, intrinsic_matrix, mask):
#     """
#     Generate a point cloud from a depth image and intrinsic matrix.
    
#     Parameters:
#     - depth_image: np.array, HxW depth image (in meters).
#     - intrinsic_matrix: np.array, 3x3 intrinsic matrix of the camera.
    
#     Returns:
#     - point_cloud: Open3D point cloud.
#     """
    
#     # Get image dimensions
#     height, width = depth_image.shape

#     # Create a meshgrid of pixel coordinates
#     u, v = np.meshgrid(np.arange(width), np.arange(height))

#     # Flatten the pixel coordinates and depth values
#     u_flat = u.flatten()
#     v_flat = v.flatten()
#     depth_flat = depth_image.flatten()
#     mask_flat = mask.flatten()

#     # Filter points where the mask is 1
#     valid_indices = np.where(mask_flat == 1)

#     # Apply the mask to the pixel coordinates and depth
#     u_valid = u_flat[valid_indices]
#     v_valid = v_flat[valid_indices]
#     depth_valid = depth_flat[valid_indices]

#     # Generate normalized pixel coordinates in homogeneous form
#     pixel_coords = np.vstack((u_valid, v_valid, np.ones_like(u_valid)))

#     # Compute inverse intrinsic matrix
#     intrinsic_inv = np.linalg.inv(intrinsic_matrix)

#     # Apply the inverse intrinsic matrix to get normalized camera coordinates
#     cam_coords = intrinsic_inv @ pixel_coords

#     # Multiply by depth to get 3D points in camera space
#     cam_coords *= depth_valid

#     # # Reshape the 3D coordinates
#     # x = cam_coords[0].reshape(height, width)
#     # y = cam_coords[1].reshape(height, width)
#     # z = depth_image

#     # # Stack the coordinates into a single 3D point array
#     # points = np.dstack((x, y, z)).reshape(-1, 3)

#     points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T

#     print("points shape: ", points.shape)

#     # remove later
#     # points = points[points[:, 2] > 0.5]
#     # print("points: ", points[:, 2])

#     # Create an Open3D point cloud object
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)

#     return point_cloud

# with open('temp_depth_img.pkl', 'rb') as f_pkl:
#     x = pickle.load(f_pkl)
#     # depth = x['depth']
#     intr = x['intr']    
#     # print("depth.shape: ", depth.shape)
#     # plt.imshow(depth)
#     # plt.show()
#     # print("intr: ", intr)

#     print(f['data/episode_00002'].keys())
#     print(f['data/episode_00002/observations/seg_semantic'])
#     seg_semantic = f['data/episode_00002/observations/seg_semantic']
#     depth = f['data/episode_00002/observations/depth']
#     plt.imshow(seg_semantic[0])
#     plt.show()
#     seg_semantic_info = np.array(f['data/episode_00002/observations_info/seg_semantic']).astype(str)

#     seq_num = 0
#     floor_id = -1
#     for seq_num in range(seg_semantic_info.shape[0]):
#         for row in seg_semantic_info[seq_num]:
#             sem_id, class_name = int(row[0]), row[1]
#             if class_name == 'floors':
#                 floor_id = sem_id
#                 break
#         print("seq_num, floor_id: ", seq_num, floor_id)

#     mask = np.zeros_like(depth[seq_num])
#     mask[seg_semantic[seq_num] != floor_id] = 1

#     plt.imshow(mask)
#     plt.show()

#     point_cloud = generate_point_cloud_from_depth(depth[seq_num], intr, mask)

#     o3d.visualization.draw_geometries([point_cloud])
# # =======================================================