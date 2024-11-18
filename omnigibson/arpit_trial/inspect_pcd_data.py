import h5py
import numpy as np

import h5py
import numpy as np
import open3d as o3d

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

    return point_cloud

def get_pcd(ep, hdf5_file):
        depth = hdf5_file[f"data/{ep}/observations/depth"]
        intr =  np.array([
            [103.8416,   0.0000,  64.0000],
            [  0.0000, 103.8416,  64.0000],
            [  0.0000,   0.0000,   1.0000]])
        
        # TODO: get extrinsic matrix from the code
        extrinsic_matrix = np.array(f["data"][first_key]["proprioceptions"]["extrinsic_matrix"])[0]
        # extrinsic_matrix = np.eye(4)
        
        # print("len(depth): ", len(depth))

        # creating mask to remove floors
        seg_semantic = hdf5_file[f'data/{ep}/observations/seg_semantic']
        seg_semantic_info = get_seg_instance_id_info(ep, hdf5_file)
        # seq_num = 0

        pcd_points = []
        pcd_normals = []
        pcd_colors = []
        for seq_num in range(len(depth)):

            # creating mask to remove floors
            floor_id = -1
            for row in seg_semantic_info[seq_num]:
                sem_id, class_name = int(row[0]), row[1]
                if class_name == 'floors':
                    floor_id = sem_id
                    break

            if floor_id != -1:
                mask = np.zeros_like(depth[seq_num])
                mask[seg_semantic[seq_num] != floor_id] = 1
            else:
                mask = np.ones_like(depth[seq_num])

            o3d_pcd = generate_point_cloud_from_depth(depth[seq_num], intr, mask, extrinsic_matrix)
            # show pcd in open3d
            # o3d.visualization.draw_geometries([o3d_pcd])
            pcd_points.append(np.asarray(o3d_pcd.points))
            pcd_colors.append(np.asarray(o3d_pcd.colors))
            pcd_normals.append(np.asarray(o3d_pcd.normals))
        
        pcd = dict()
        pcd['points'] = np.array(pcd_points)
        pcd['colors'] = np.array(pcd_colors)
        pcd['normals'] = np.array(pcd_normals)
        return pcd

# class VisualizerWithCallback:
#     def __init__(self, point_cloud):
#         self.vis = o3d.visualization.VisualizerWithEditing()
#         self.pcd = point_cloud
#         self.points = np.asarray(self.pcd.points)
        
#     def run(self):
#         self.vis.create_window()
#         # self.vis.add_geometry(self.pcd)
        
#         # Set camera parameters
#         view_control = self.vis.get_view_control()
#         view_control.change_field_of_view(60.0)
#         view_control.set_zoom(0.7)
#         view_control.set_front([0, 0, -1])
#         view_control.set_lookat([0, 0, 0])
#         view_control.set_up([0, -1, 0])
        
#         # Run visualizer and get picked points
#         picked_points = self.vis.run()  # Returns indices of picked points
        
#         # Print coordinates of picked points
#         if picked_points is not None:
#             for idx in picked_points:
#                 if idx < len(self.points):
#                     point = self.points[idx]
#                     print(f"Selected point {idx}: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}")
        
#         self.vis.destroy_window()

# def visualize_pointcloud_and_action(points, colors, action=None, eef_pos=None):
#     # Create point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     # Add colors to point cloud
#     if colors is not None:
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#     else:
#         # If no colors provided, use uniform gray color for visibility
#         pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
#     # Create visualizer instance
#     vis_obj = VisualizerWithCallback(pcd)
    
#     if action is not None:
#         # Create line geometry for action vector starting from eef_pos
#         start_point = eef_pos
#         end_point = eef_pos + action
#         points = [start_point, end_point]
#         lines = [[0, 1]]
#         colors = [[1, 0, 0]]  # Red color for action vector
        
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(points)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         # Add a sphere at the start point for better visibility
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50.0)
#         sphere.translate(start_point)
#         sphere.paint_uniform_color([0, 1, 0])  # Green color for start point
        
#         vis_obj.vis.add_geometry(line_set)
#         vis_obj.vis.add_geometry(sphere)

#         # #  Create line geometry for action vector
#         # points = [[0, 0, 0], action]
#         # lines = [[0, 1]]
#         # colors = [[1, 0, 0]]
        
#         # line_set = o3d.geometry.LineSet()
#         # line_set.points = o3d.utility.Vector3dVector(points)
#         # line_set.lines = o3d.utility.Vector2iVector(lines)
#         # line_set.colors = o3d.utility.Vector3dVector(colors)
        
#         # vis_obj.vis.add_geometry(line_set)
    
#     # Add coordinate frame
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     vis_obj.vis.add_geometry(coord_frame)
    
#     # Run visualizer
#     vis_obj.run()

def visualize_pointcloud_and_action(points1, colors1, points2, colors2, action=None, eef_pos=None):
    # Create visualizer instance
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Create first point cloud object
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    if colors1 is not None:
        pcd1.colors = o3d.utility.Vector3dVector(colors1)
    else:
        pcd1.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd1)
    
    offset_y = -2.0
    # Create second point cloud object (offset in x direction)
    pcd2 = o3d.geometry.PointCloud()
    points2_offset = points2.copy()
    points2_offset[:, 1] += offset_y  # Offset in x direction
    pcd2.points = o3d.utility.Vector3dVector(points2_offset)
    if colors2 is not None:
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
    else:
        pcd2.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd2)
    
    
    if action is not None:
        # Create cylinder for action vector
        start_point = eef_pos
        end_point = eef_pos + action  # Scale action vector for better visibility
        print("start_point: ", start_point)
        print("end_point: ", end_point)

        # Calculate cylinder parameters
        vector = end_point - start_point
        length = np.linalg.norm(vector)
        direction = vector / length
        
        # Create cylinder (oriented along z-axis by default)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=length)
        
        # Calculate rotation to align cylinder with action vector
        # Default cylinder direction is [0, 0, 1]
        default_direction = np.array([0, 0, 1])
        # Find rotation axis and angle
        rotation_axis = np.cross(default_direction, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 0:  # if not parallel
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.dot(default_direction, direction))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            cylinder.rotate(R, center=[0, 0, 0])
        
        # Move cylinder to correct position
        cylinder.translate(start_point + vector/2)
        cylinder.paint_uniform_color([1, 0, 0])  # Red color
        vis.add_geometry(cylinder)

        # Add a sphere at the start point for better visibility
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(start_point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color for start point
        vis.add_geometry(sphere)
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coord_frame)
    
    # Set camera parameters
    view_control = vis.get_view_control()
    view_control.change_field_of_view(60.0)
    # view_control.set_zoom(0.7)
    view_control.set_zoom(1.5)
    # view_control.set_front([0, 0, -1])
    view_control.set_front([-0.013842371518927768, 0.5269709674468008, 0.8497705503363334])
    # view_control.set_lookat([0, 0, 0])
    view_control.set_lookat([1.0, 1.0, 1.0])
    # view_control.set_up([0, -1, 0])
    view_control.set_up([0.9690995313503652, 0.2163931972079168, -0.11840642946495106])
    
    # Run visualizer
    vis.run()

    # # Get and print camera parameters before destroying window
    # view_control = vis.get_view_control()
    # cam = view_control.convert_to_pinhole_camera_parameters()
    # # Get zoom
    # # zoom = view_control.get_zoom()
    # # Get view matrix
    # view_matrix = np.array(cam.extrinsic)
    # # Calculate front, up, and lookat
    # front = -view_matrix[:3, 2]  # negative z-axis of camera coordinate system
    # up = -view_matrix[:3, 1]     # negative y-axis of camera coordinate system
    # # lookat = np.array(view_control.get_lookat())
    # print("\nCamera parameters:")
    # # print(f"zoom = {zoom}")
    # print(f"front = {front.tolist()}")
    # print(f"up = {up.tolist()}")
    # # print(f"lookat = {lookat.tolist()}")

    vis.destroy_window()

# Read and visualize data
with h5py.File("/home/arpit/test_projects/OmniGibson/temp/dataset.hdf5", "r") as f:
    first_key = list(f["data"].keys())[2]
    extr = np.array(f["data"][first_key]["proprioceptions"]["extrinsic_matrix"])[2]
    print("extr: ", extr)
    # breakpoint()
    pcd = get_pcd(first_key, f)
    # breakpoint()
    point_clouds = pcd['points']
    point_colors = pcd['colors']
    actions = np.array(f["data"][first_key]["actions"]["actions"])[:, 3:6]
    eef_pos = np.array(f["data"][first_key]["proprioceptions"]["right_eef_pos"])
    # actions_to_visualize = actions + eef_pos[:-1]
    print("actions: ", actions.shape)
    # actions = np.array([[0.0, 0.0, 0.0]])

    visualize_pointcloud_and_action(point_clouds[0], point_colors[0], point_clouds[1], point_colors[1], action=actions[0], eef_pos=eef_pos[0])