import h5py
import numpy as np
import pickle
np.set_printoptions(suppress=True, precision=3)

# f = h5py.File('moma_pick_and_place/dataset.hdf5', "r")
# # ======================== Basic hdf5 testing ======================
# # f = h5py.File('prior/dataset.hdf5', "r")
# print("len: ", len(f['data']))
# print("--", f['data/episode_00000/observations'].keys())
# print("--", np.array(f['data/episode_00000/observations/rgb']).shape)
# print(np.array(f['data/episode_00000/actions/actions']))
# print("--", np.array(f[f'data/episode_00000/observations_info']['seg_semantic']))
# print("joint efforts: ", np.array(f[f'data/episode_00000/proprioceptions/joint_qeffort']).shape)
# print("--", np.array(f[f'data/episode_00001/observations_info']['seg_instance_id_strings']))
# for ep in f['data'].keys():
#     print(ep, np.array(f['data'][f'{ep}/actions/actions']).shape)
#     print("ep: ", ep)
#     print("--", np.array(f[f'data/{ep}/observations_info']['seg_instance_id']))
# obs_info = f['data/episode_00000/observations_info'].keys()
# print("obs_info: ", obs_info)
# grasped_state = np.array(f['data/episode_00000/extras/grasps'])
# print("grasped_state: ", grasped_state)
# img = np.array(f[f'data/episode_00300/observations/gripper_obj_seg'])
# print("img.shape: ", img.shape)
# fig, ax = plt.subplots(1, img.shape[0])
# for i in range(len(img)):
#     ax[i].imshow(img[i])
# plt.show()
# ===================================================================


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
    

# ============= checking point cloud ====================
import open3d as o3d

def generate_point_cloud_from_depth(depth_image, intrinsic_matrix):
    """
    Generate a point cloud from a depth image and intrinsic matrix.
    
    Parameters:
    - depth_image: np.array, HxW depth image (in meters).
    - intrinsic_matrix: np.array, 3x3 intrinsic matrix of the camera.
    
    Returns:
    - point_cloud: Open3D point cloud.
    """
    
    # Get image dimensions
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the pixel coordinates and depth values
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_image.flatten()

    # Generate normalized pixel coordinates in homogeneous form
    pixel_coords = np.vstack((u_flat, v_flat, np.ones_like(u_flat)))

    # Compute inverse intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    # Apply the inverse intrinsic matrix to get normalized camera coordinates
    cam_coords = intrinsic_inv @ pixel_coords

    # Multiply by depth to get 3D points in camera space
    cam_coords *= depth_flat

    # Reshape the 3D coordinates
    x = cam_coords[0].reshape(height, width)
    y = cam_coords[1].reshape(height, width)
    z = depth_image

    # Stack the coordinates into a single 3D point array
    points = np.dstack((x, y, z)).reshape(-1, 3)

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

with open('temp_depth_img.pkl', 'rb') as f:
    x = pickle.load(f)
    depth = x['depth']
    intr = x['intr']
    point_cloud = generate_point_cloud_from_depth(depth, intr)
    o3d.visualization.draw_geometries([point_cloud])
# =======================================================