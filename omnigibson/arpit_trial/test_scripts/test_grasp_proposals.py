import h5py
import random
import os

import torch as th  
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import grasp_pose_generator as gpg

from grasp_selector import GraspSelector
from sklearn_extra.cluster import KMedoids
from scipy.linalg import logm, norm

# Function to compute geodesic distance between two rotation matrices
def geodesic_distance(R1, R2):
    """Computes geodesic distance between two 3x3 rotation matrices."""
    # Extract the z-axis component of the rotation matrices
    z_axis_component_R1 = R1[:, 2]
    z_axis_component_R2 = R2[:, 2]
    
    # Compute the dot product of the z-axis components to get the cosine of the angle between them
    dot_product = np.dot(z_axis_component_R1, z_axis_component_R2)
    
    # Compute the angle between the z-axis components
    angle_between_z_axes = np.arccos(dot_product)
    if np.isnan(angle_between_z_axes):
        return 5.0
    print("angle_between_z_axes: ", angle_between_z_axes)
    
    # Return the angle as the distance along the z-axis
    return angle_between_z_axes
    
    # relative_rotation = R1.T @ R2
    # log_rot = logm(relative_rotation)  # Matrix logarithm
    # return norm(log_rot, 'fro') / np.sqrt(2)

# Custom distance function for pose clustering
def pose_distance(pose1, pose2, translation_weight=1.0, rotation_weight=1.0):
    """Combines translation and rotation distances for two 4x4 pose matrices."""
    # Extract rotation matrices and translations
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    
    # Compute geodesic distance for rotation
    rotation_dist = geodesic_distance(R1, R2)
    
    # Compute Euclidean distance for translation
    translation_dist = np.linalg.norm(t1 - t2)
    # print("translation_dis, rotation_dist: ", translation_dist, rotation_dist)
    
    # Weight the rotation and translation distances (you can adjust the weights)
    # w1 = 1.0
    # w2 = 2.0
    return (rotation_weight * rotation_dist) + (translation_weight * translation_dist)

def cluster_sampled_grasps(grasp_poses, k=3, translation_weight=1.0, rotation_weight=1.0):
    # ============== trial 1 ===================
    # Flatten the poses into a list of matrices (needed for the distance matrix calculation)
    n_poses = len(grasp_poses)
    distance_matrix = np.zeros((n_poses, n_poses))

    # Compute the pairwise distance matrix
    for i in range(n_poses):
        for j in range(i, n_poses):
            dist = pose_distance(grasp_poses[i], grasp_poses[j], translation_weight, rotation_weight)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric

    # Perform K-medoids clustering with the custom distance matrix
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)

    # Print the resulting cluster labels for each grasp pose
    print("Cluster labels for grasp poses:", labels)
    medoid_indices = kmedoids.medoid_indices_
    print("medoid_indices: ", medoid_indices)
    # comment/uncomment this
    return grasp_poses[medoid_indices]
    # ==========================================

def random_point_dropout(point_cloud, fraction_to_keep=0.1):
    indices = random.sample(range(len(point_cloud.points)), int(len(point_cloud.points) * fraction_to_keep))
    downsampled_point_cloud = point_cloud.select_by_index(indices)
    return downsampled_point_cloud


def generate_point_cloud_from_depth(depth_image, intrinsic_matrix, mask, extrinsic_matrix):
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
    mask_flat = mask.flatten()

    # # Filter points where the mask is 1
    # valid_indices = np.where(mask_flat == 1)
    
    # Filter points where the mask is 1 AND depth is valid (not inf and not 0)
    valid_indices = np.where(
        (mask_flat == 1) & 
        (np.isfinite(depth_flat)) &  # Remove inf values
        (depth_flat > 0)             # Remove 0 or negative values
    )[0]

    # Apply the mask to the pixel coordinates and depth
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    depth_valid = depth_flat[valid_indices]

    # Generate normalized pixel coordinates in homogeneous form
    pixel_coords = np.vstack((u_valid, v_valid, np.ones_like(u_valid)))

    # Compute inverse intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    # Apply the inverse intrinsic matrix to get normalized camera coordinates
    cam_coords = intrinsic_inv @ pixel_coords

    # Multiply by depth to get 3D points in camera space
    cam_coords *= depth_valid
    # breakpoint()

    # # Reshape the 3D coordinates
    # x = cam_coords[0].reshape(height, width)
    # y = cam_coords[1].reshape(height, width)
    # z = depth_image

    # # Stack the coordinates into a single 3D point array
    # points = np.dstack((x, y, z)).reshape(-1, 3)

    points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T
    print("points shape: ", points.shape)

    # transform points to world frame
    # make points homogeneous
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = extrinsic_matrix @ points.T
    points = points.T
    # remove homogeneous coordinate
    points = points[:, :3]

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Estimate normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud.orient_normals_consistent_tangent_plane(k=30)
    # camera_location = np.array([10000, 10000, 10000])
    # point_cloud.orient_normals_towards_camera_location(camera_location)
    # point_cloud.orient_normals_towards_camera_location()

    print("point_cloud: ", np.array(point_cloud.points).shape)
    # unique_points = np.unique(point_cloud.points, axis=0)
    # print("Unique points in point_cloud: ", unique_points.shape)
    point_cloud = random_point_dropout(point_cloud, fraction_to_keep=0.3)
    print("point_cloud: ", np.array(point_cloud.points).shape)

    return point_cloud

def extract_observations_info_from_hdf5(obs_info_strings, obs_info_shapes):
    # Reconstruct original structure
    idx = 0
    reconstructed_data = []
    for shape in obs_info_shapes:
        sublist = []
        for _ in range(shape):
            sublist.append(list(map(lambda x: x.decode('utf-8'), obs_info_strings[idx:idx+2])))
            idx += 2
        reconstructed_data.append(sublist)

    reconstructed_data = np.array(reconstructed_data, dtype=object)
    # for i in range(len(reconstructed_data)):
    #     print(i, np.array(reconstructed_data[i]).shape)
    #     if i == 0:
    #         print(reconstructed_data)
    return reconstructed_data
    
def get_seg_semantic_info(ep, hdf5_file):
    # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
    if 'seg_semantic_strings' in hdf5_file[f'data/{ep}/observations_info'].keys():
        seg_semantic_strings = np.array(hdf5_file["data/{}/observations_info/seg_semantic_strings".format(ep)])
        seg_semantic_shapes = np.array(hdf5_file["data/{}/observations_info/seg_semantic_shapes".format(ep)])
        seg_semantic = extract_observations_info_from_hdf5(obs_info_strings=seg_semantic_strings, 
                                                                    obs_info_shapes=seg_semantic_shapes)
        # print("111: ", seg_semantic.shape)
    else:
        hd5key = "data/{}/observations_info/seg_semantic".format(ep)
        # seg_semantic = hdf5_file[hd5key]
        seg_semantic = np.array(hdf5_file[hd5key]).astype(str)
        # print("222: ", seg_semantic.shape)
    return seg_semantic

def get_seg_instance_info(ep, hdf5_file):
    # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
    if 'seg_instance_strings' in hdf5_file[f'data/{ep}/observations_info'].keys():
        seg_instance_strings = np.array(hdf5_file["data/{}/observations_info/seg_instance_strings".format(ep)])
        seg_instance_shapes = np.array(hdf5_file["data/{}/observations_info/seg_instance_shapes".format(ep)])
        seg_instance = extract_observations_info_from_hdf5(obs_info_strings=seg_instance_strings, 
                                                                    obs_info_shapes=seg_instance_shapes)
        # print("111: ", seg_instance.shape)
    else:
        hd5key = "data/{}/observations_info/seg_instance".format(ep)
        # seg_instance = hdf5_file[hd5key]
        seg_instance = np.array(hdf5_file[hd5key]).astype(str)
        # print("222: ", seg_instance.shape)
    return seg_instance

def get_pcd(ep, hdf5_file, waypoint=-1, obj_name=None):
    depth = hdf5_file[f"data/{ep}/observations/depth"][waypoint:]
    intr =  np.array([
        [103.8416,   0.0000,  64.0000],
        [  0.0000, 103.8416,  64.0000],
        [  0.0000,   0.0000,   1.0000]])
    
    # TODO: get extrinsic matrix from the code
    extrinsic_matrix = np.array(hdf5_file["data"][ep]["proprioceptions"]["extrinsic_matrix"])[0]
    # extrinsic_matrix = np.eye(4)
    
    # print("len(depth): ", len(depth))

    # creating mask to remove floors
    seg_semantic = hdf5_file[f'data/{ep}/observations/seg_semantic'][waypoint:]
    seg_instance = hdf5_file[f'data/{ep}/observations/seg_instance'][waypoint:]
    
    # Change here
    # seg_semantic_info = get_seg_semantic_info(ep, hdf5_file)
    seg_instance_info = get_seg_instance_info(ep, hdf5_file)[waypoint:]
    
    pcd_points = []
    pcd_normals = []
    pcd_colors = []
    for seq_num in range(len(depth)):

        # creating mask to keep object
        obj_id = -1
        # Change here
        # for row in seg_semantic_info[seq_num]:
        for row in seg_instance_info[seq_num]:
            sem_id, class_name = int(row[0]), row[1]
            # Change here
            if class_name == obj_name:
                obj_id = sem_id
                break

        breakpoint()
        if obj_id != -1:
            mask = np.zeros_like(depth[seq_num])
            # Change here
            # mask[seg_semantic[seq_num] != floor_id] = 1
            mask[seg_instance[seq_num] == obj_id] = 1
        else:
            mask = np.ones_like(depth[seq_num])
        # mask = np.ones_like(depth[seq_num])

        o3d_pcd = generate_point_cloud_from_depth(depth[seq_num], intr, mask, extrinsic_matrix)

        # show pcd in open3d
        o3d.visualization.draw_geometries([o3d_pcd],  point_show_normal=True)
        pcd_points.append(np.asarray(o3d_pcd.points))
        pcd_colors.append(np.asarray(o3d_pcd.colors))
        pcd_normals.append(np.asarray(o3d_pcd.normals))
    
    return o3d_pcd

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    

def main():
    set_all_seeds(seed=0)
    hdf5_file = h5py.File("/home/arpit/projects/OmniGibson/place_in_shelf_temp/dataset.hdf5", "r")
    ep = "episode_00001"
    # obj_name = "box"
    obj_name = "can_of_baking_mix"
    test_cloud_with_normals = get_pcd(ep=ep, hdf5_file=hdf5_file, obj_name=obj_name)
    num_samples = len(test_cloud_with_normals.points)
    print("num_samples: ", num_samples)
    
    # test_cloud_file = "/home/arpit/projects/OmniGibson/DoorCIP.ply"
    # test_cloud_with_normals = o3d.io.read_point_cloud(test_cloud_file)
    # world_frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    door_frame = np.eye(4)
    # door_frame[:3, :3] = np.array([\
    #                                 [0, 0, 1], \
    #                                 [0, 1, 0], \
    #                                 [-1, 0, 0]])
    # door_frame[:3, 3] = [0.14, 0.348, 0.415]

    gs = GraspSelector(door_frame, test_cloud_with_normals)

    sampled_poses = gs.getRankedGraspPoses()
    print("sampled_poses: ", np.array(sampled_poses).shape)
    desired_sampled_poses = sampled_poses[:num_samples]
    desired_sampled_poses = [gpg.translateFrameNegativeZ(p, gs.dist_from_point_to_ee_link) for p in desired_sampled_poses]

    desired_sampled_poses = cluster_sampled_grasps(np.array(desired_sampled_poses),
                                                     k=2,
                                                     translation_weight=1.0,
                                                     rotation_weight=1.0)
    
    for i in range(len(desired_sampled_poses)):
        print("desired_sampled_poses: ", desired_sampled_poses[i])

    # breakpoint()

    gs.visualizeGraspPoses(desired_sampled_poses)


if __name__ == '__main__':
    main()