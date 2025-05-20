import random
import h5py
import open3d as o3d
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(suppress=True, precision=3)

from pointnet2.data_utils.utils import obtain_mask_by_removing_floor

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

    # breakpoint()
    points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T
    # points = np.vstack((cam_coords[0], -depth_valid, -cam_coords[1])).T
    # points = np.vstack((-cam_coords[1], -depth_valid, cam_coords[0])).T

    if points.shape[0] == 0:
        print("1111111111111111111111")
        points = np.zeros((1, 3))

    # pad points so that the total number of points are 128*128
    target_size=(height*width, 3)
    N_i = points.shape[0]
    pad_rows = target_size[0] - N_i
    padding = ((0, pad_rows), (0, 0))
    points = np.pad(points, padding, mode='edge')

    # print("points shape: ", points.shape)

    # remove later
    # points = points[points[:, 2] > 0.5]
    # print("points: ", points[:, 2])


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

    # compute normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

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

def get_seg_instance_id_info(ep, hdf5_file):
    # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id_id
    if 'seg_instance_id_strings' in hdf5_file[f'{ep}/observations_info'].keys():
        seg_instance_id_strings = np.array(hdf5_file["{}/observations_info/seg_instance_id_strings".format(ep)])
        seg_instance_id_shapes = np.array(hdf5_file["{}/observations_info/seg_instance_id_shapes".format(ep)])
        seg_instance_id = extract_observations_info_from_hdf5(obs_info_strings=seg_instance_id_strings, 
                                                                    obs_info_shapes=seg_instance_id_shapes)
        # print("111: ", seg_instance_id.shape)
    else:
        hd5key = "{}/observations_info/seg_instance_id".format(ep)
        # seg_instance_id = hdf5_file[hd5key]
        seg_instance_id = np.array(hdf5_file[hd5key]).astype(str)
        # print("222: ", seg_instance_id.shape)
    return seg_instance_id

def obtain_pcd(hdf5_file, ep, seq_num):
    seg_instance_id = hdf5_file[f"{ep}/observations/seg_instance_id"]
    seg_instance_id_info = get_seg_instance_id_info(ep, hdf5_file)
    depth = hdf5_file[f"{ep}/observations/depth"]
    intr =  np.array([
            [103.8416,   0.0000,  64.0000],
            [  0.0000, 103.8416,  64.0000],
            [  0.0000,   0.0000,   1.0000]])
        
    extrinsic_matrix = np.array(hdf5_file[ep]["proprioceptions"]["extrinsic_matrix"])[0]
    
    # mask = obtain_mask_by_removing_floor
    mask = obtain_mask_by_removing_floor(depth, seg_instance_id, seg_instance_id_info, seq_num)

    o3d_pcd = generate_point_cloud_from_depth(depth[seq_num], intr, mask, extrinsic_matrix)
    # o3d.visualization.draw_geometries([o3d_pcd])
    points = np.asarray(o3d_pcd.points)
    normals = np.asarray(o3d_pcd.normals)
    
    # # Randomly drop points from the point cloud
    # keep_ratio = 0.6  # Keep x% of points randomly
    # num_points = points.shape[0]
    # mask = np.random.choice([True, False], size=num_points, p=[keep_ratio, 1-keep_ratio])
    # print("11 points, normals: ", points.shape, normals.shape)
    # points = points[mask]
    # normals = normals[mask]
    # print("22 points, normals: ", points.shape, normals.shape)

    pcd = np.concatenate((points, normals), axis=1)
    return pcd

def partition_dataset_train_valid(dataset):
    np.random.seed(0)
    random.seed(0)
    hdf5_file_path = dataset
    file = h5py.File(hdf5_file_path, 'a')
    all_keys = list(file['data'].keys())

    if 'mask' in file:
        print("deleting")
        input("Press enter to delete")
        del file['mask']
        
    print("file.keys: ", file.keys())

    # Shuffle the keys
    random.shuffle(all_keys)
    print("all_keys: ", all_keys)

    # Calculate the number of test samples (10% of the total data)
    num_test_samples = min(int(len(all_keys) * 0.1), 75)

    # Split the keys into train and test sets
    test_keys = all_keys[:num_test_samples]
    test_keys = [np.bytes_(s) for s in test_keys]
    train_keys = all_keys[num_test_samples:]
    train_keys = [np.bytes_(s) for s in train_keys]

    # Print the results
    print(f"Train keys: {len(train_keys)}")
    print(f"Test keys: {len(test_keys)}")
    
    file = h5py.File(hdf5_file_path, 'a')
    group_key = 'mask'
    if group_key not in file:
        group = file.create_group(group_key)
    group = file[group_key]

    group.create_dataset('train', data=train_keys, compression='gzip', compression_opts=9)
    group.create_dataset('valid', data=test_keys, compression='gzip', compression_opts=9)


# Utilitiy functions for copying / editing HDF5 files
        
def copy_attrs(src, dst):
    """ Copy attributes from one HDF5 object to another """
    for key, value in src.attrs.items():
        dst.attrs[key] = value

def copy_group(source_group, dest_group):
    for key in source_group:
        item = source_group[key]
        # print("key, type(item): ", key, type(item))
        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(key)
            copy_attrs(item, new_group)
            copy_group(item, new_group)

        elif isinstance(item, h5py.Dataset):
            dest_group.create_dataset(key, data=item[()], compression='gzip', compression_opts=9)
            copy_attrs(item, dest_group)

def edit_and_merge_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    group = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    if 'data' in old_file.keys():
        old_file = old_file['data']
    copy_group(old_file, group) 

def reame_ep_and_merge_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    new_file = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    if 'data' in old_file.keys():
        old_file = old_file['data']

    # find the last episode name in the destination hdf5 file
    counter = len(new_file)
    # print("counter: ", counter)

    for i, org_ep_name in enumerate(old_file):
        # New episode name in the destination hdf5 file
        new_ep_name = f'episode_{counter:05d}'
        # print("i, new_ep_name: ", i, new_ep_name)
        # Create a new group with new episode name in the destination hdf5 file
        new_group = new_file.create_group(new_ep_name)
        copy_group(old_file[org_ep_name], new_group) 
        counter += 1

def create_new_hdf5_from_old_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    new_file = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    if 'data' in old_file.keys():
        old_file = old_file['data']

    # add some attributes
    env_args = '{\n    "env_name": "OG",\n    "env_kwargs": {\n        "robots": "Panda",\n        "controller_configs": {\n            "type": "OSC_POSITION",\n            "input_max": 1,\n            "input_min": -1,\n            "output_max": [\n                0.05,\n                0.05,\n                0.05\n            ],\n            "output_min": [\n                -0.05,\n                -0.05,\n                -0.05\n            ],\n            "kp": 150,\n            "damping_ratio": 1,\n            "impedance_mode": "fixed",\n            "kp_limits": [\n                0,\n                300\n            ],\n            "damping_ratio_limits": [\n                0,\n                10\n            ],\n            "position_limits": null,\n            "control_delta": true,\n            "interpolation": null,\n            "ramp_ratio": 0.2\n        },\n        "has_renderer": false,\n        "has_offscreen_renderer": false,\n        "camera_names": [\n            "agentview",\n            "robot0_eye_in_hand",\n            "sideview"\n        ],\n        "ignore_done": true,\n        "use_camera_obs": true,\n        "control_freq": 10,\n        "image_size": [\n            128,\n            128\n        ]\n    },\n    "type": 1\n}'
    new_file.attrs["env_args"] = env_args

    counter = 0
    for i, org_ep_name in enumerate(old_file):
        rgb = []
        depth = []
        actions = []
        dones = []
        rewards = []
        pcds = []
        

        # check if there is a collision
        for seq_num in range(len(old_file[f"{org_ep_name}/actions/actions"])):
            if not np.array(old_file[f"{org_ep_name}/extras/contacts"])[seq_num+1]:
                pcd = obtain_pcd(old_file, org_ep_name, seq_num=seq_num)
                pcds.append(pcd)
                rgb.append(np.array(old_file[f"{org_ep_name}/observations/rgb"])[seq_num, :, :, :3])
                depth.append(np.array(old_file[f"{org_ep_name}/observations/depth"])[seq_num])
                old_action = np.array(old_file[f"{org_ep_name}/actions/actions"])[seq_num]
                new_action = np.concatenate((old_action[3:6], old_action[-1:]))
                actions.append(new_action)
                rewards.append(0)
                # print("len(rgb): ", len(rgb))
                counter += 1
        for _ in range(len(rgb) - 1):
            dones.append(False)
        dones.append(True)

        pcds = np.array(pcds)
        # print("pcds.shape: ", pcds.shape)
        if pcds.shape[0] == 0:
            continue

        # create the episode group
        new_ep_name = f'demo_{i}'
        new_file.create_group(new_ep_name)
        
        # add to new hdf5 file
        new_file[new_ep_name].create_group('obs')
        new_file[new_ep_name].create_dataset('obs/rgb', data=rgb, compression='gzip', compression_opts=9)
        new_file[new_ep_name].create_dataset('obs/depth', data=depth, compression='gzip', compression_opts=9)
        new_file[new_ep_name].create_dataset('obs/pcd', data=pcds, compression='gzip', compression_opts=9)
        new_file[new_ep_name].create_dataset('actions', data=actions, compression='gzip', compression_opts=9)
        new_file[new_ep_name].create_dataset('dones', data=dones, compression='gzip', compression_opts=9)
        new_file[new_ep_name].create_dataset('rewards', data=rewards, compression='gzip', compression_opts=9)
        new_file[new_ep_name].attrs["num_samples"] = len(actions)

        new_file.attrs["total"] = counter


def modify_dataset_in_hdf5_file(file):
    for i, ep in enumerate(file['data'].keys()):
        # Followed: https://stackoverflow.com/questions/22922584/how-to-overwrite-array-inside-h5-file-using-h5py
        actions = file[f'data/{ep}/actions/actions']
        del file[f'data/{ep}/actions/actions']
        seq_len = actions.shape[0]
        zeros = np.zeros((seq_len, 3))
        actions = np.hstack((zeros, actions))
        file.create_dataset(f'data/{ep}/actions/actions', data=actions)

def main():
    # hdf5_file = '/home/arpit/test_projects/OmniGibson/pick_data/dataset.hdf5'
    # file = h5py.File(hdf5_file, 'a')
    # modify_dataset_in_hdf5_file(file)
    # reame_ep_and_merge_hdf5('/home/arpit/projects/OmniGibson/place_in_drawer_obj_dropping/dataset.hdf5', '/home/arpit/projects/OmniGibson/obj_dropping/dataset.hdf5')
    partition_dataset_train_valid('/home/arpit/projects/OmniGibson/place_in_shelf/place_in_shelf_robomimic_dataset.hdf5')

    # create new hdf5 file based on an old one with keys etc. changed
    # create_new_hdf5_from_old_hdf5('/home/arpit/projects/OmniGibson/place_in_shelf/dataset.hdf5', '/home/arpit/projects/OmniGibson/place_in_shelf/place_in_shelf_robomimic_dataset.hdf5')

if __name__ == "__main__":
    main()