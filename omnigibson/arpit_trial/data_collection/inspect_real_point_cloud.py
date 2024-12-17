import h5py
import numpy as np
import torch

import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from pointnet2.models import action_pointnet2_cls_ssg
from omnigibson.arpit_trial.test_sam import obtain_mask

def predict_collisions(points, actions):
    points = torch.from_numpy(np.array(points))
    # actions = actions.unsqueeze(0)
    actions = torch.from_numpy(np.array(actions)).to(dtype=torch.float64)
        
    # remove later
    votes = 10
    actions_tile = actions.repeat(votes, 1)
    pos_noise = torch.empty(votes, 3).uniform_(-0.005, 0.005)
    actions_tile[:, 3:6] = actions_tile[:, 3:6] + pos_noise

    points, actions = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda()
    actions_tile = actions_tile.type(torch.FloatTensor).cuda()
    points = points.transpose(2, 1)
    
    # ---------------------------
    vote_num = 1    
    for _ in range(vote_num):
        pred, _ = classifier(points, actions)
        # vote_pool += pred
    # pred = vote_pool / vote_num
    # pred_choice = pred.data.max(1)[1]

    probabilities = torch.sigmoid(pred)
    # Convert probabilities to binary predictions (0 or 1)
    pred_choice = (probabilities >= 0.5).float()

    print(f"pred_choice, pred_prob: ", pred_choice.item(), probabilities.item())
    # -----------------------------

    return pred_choice.item(), probabilities.item()

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
    # breakpoint()
    points = points / 1000.0    
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = extrinsic_matrix @ points.T
    points = points.T
    # remove homogeneous coordinate
    points = points[:, :3]

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

def get_pcd(data_dict):
        depth = data_dict["depth"].squeeze()
        intr = data_dict["cam_intr"]
        
        extrinsic_matrix = data_dict["cam_extr"]
        # extrinsic_matrix = np.eye(4)

        # breakpoint()
        
        pcd_points = []
        pcd_normals = []
        pcd_colors = []
        mask = obtain_mask()

        # breakpoint()

        o3d_pcd = generate_point_cloud_from_depth(depth, intr, mask, extrinsic_matrix)
        
        # show pcd in open3d
        # o3d.visualization.draw_geometries([o3d_pcd])
        
        pcd_points.append(np.asarray(o3d_pcd.points))
        pcd_colors.append(np.asarray(o3d_pcd.colors))
        pcd_normals.append(np.asarray(o3d_pcd.normals))

        # remove points that are too close to the floor
        breakpoint()
        pcd_points[-1] = pcd_points[-1][pcd_points[-1][:, 2] > 0.15]
        # pcd_colors[-1] = pcd_colors[-1][pcd_points[-1][:, 2] > 0.15]
        # pcd_normals[-1] = pcd_normals[-1][pcd_points[-1][:, 2] > 0.15]

        # Randomly drop points from the point cloud
        keep_ratio = 0.7  # Keep 50% of points randomly
        num_points = len(pcd_points[-1])
        mask = np.random.choice([True, False], size=num_points, p=[keep_ratio, 1-keep_ratio])
        
        pcd_points[-1] = pcd_points[-1][mask]
        # pcd_colors[-1] = pcd_colors[-1][mask] if len(pcd_colors[-1]) > 0 else pcd_colors[-1]
        # pcd_normals[-1] = pcd_normals[-1][mask] if len(pcd_normals[-1]) > 0 else pcd_normals[-1]
        
        pcd = dict()
        pcd['points'] = np.array(pcd_points)
        pcd['colors'] = np.array(pcd_colors)
        pcd['normals'] = np.array(pcd_normals)
        return pcd

class VisualizerWithCallback:
    def __init__(self, point_cloud):
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.pcd = point_cloud
        self.points = np.asarray(self.pcd.points)
        
    def run(self):
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        
        # Set camera parameters
        view_control = self.vis.get_view_control()
        view_control.change_field_of_view(60.0)
        view_control.set_zoom(0.7)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        
        # Run visualizer and get picked points
        picked_points = self.vis.run()  # Returns indices of picked points
        
        # Print coordinates of picked points
        if picked_points is not None:
            for idx in picked_points:
                if idx < len(self.points):
                    point = self.points[idx]
                    print(f"Selected point {idx}: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}")
        
        self.vis.destroy_window()

def visualize_pointcloud_and_action_pt_selection(points, colors=None, action=None, eef_pos=None):
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors to point cloud
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # If no colors provided, use uniform gray color for visibility
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Create visualizer instance
    vis_obj = VisualizerWithCallback(pcd)
    
    if action is not None:
        # Create line geometry for action vector starting from eef_pos
        start_point = eef_pos
        end_point = eef_pos + action
        points = [start_point, end_point]
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # Red color for action vector
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Add a sphere at the start point for better visibility
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50.0)
        sphere.translate(start_point)
        sphere.paint_uniform_color([0, 1, 0])  # Green color for start point
        
        vis_obj.vis.add_geometry(line_set)
        vis_obj.vis.add_geometry(sphere)

        # #  Create line geometry for action vector
        # points = [[0, 0, 0], action]
        # lines = [[0, 1]]
        # colors = [[1, 0, 0]]
        
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(points)
        # line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # vis_obj.vis.add_geometry(line_set)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_obj.vis.add_geometry(coord_frame)
    
    # Run visualizer
    vis_obj.run()

def visualize_pointcloud_and_action(points1, colors1, points2, colors2, action=None, eef_pos=None):
    # Create visualizer instance
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # # Create first point cloud object
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(points1)
    # if colors1 is not None:
    #     pcd1.colors = o3d.utility.Vector3dVector(colors1)
    # else:
    #     pcd1.paint_uniform_color([0.7, 0.7, 0.7])
    # vis.add_geometry(pcd1)
    
    # offset_y = -2.0
    offset_y = 0.0
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
        length = np.linalg.norm(vector) + 0.000001
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


import pickle
# f_name = "0001.pickle"
# f = open(f"/home/arpit/test_projects/OmniGibson/real_world_data/{f_name}", "rb")
# data_dict = pickle.load(f)
# for k in data_dict.keys():
#     print("k, v: ", k, np.array(data_dict[k]).shape)

# # breakpoint()
# pcd = get_pcd(data_dict)

# # breakpoint()

# waypoint_number = 0
# point_clouds = pcd['points']

# # save pcd to pickle file
# with open('real_world_data/pcd_0001.pickle', 'wb') as f:
#     pickle.dump(point_clouds, f)

# visualize_pointcloud_and_action_pt_selection(point_clouds[waypoint_number])


# ------------------------------------------------------------------------------------------------
num_class = 1
experiment_dir = "/home/arpit/test_projects/Pointnet_Pointnet2_pytorch/pointnet2/log/classification/run_place_new_data"

classifier = action_pointnet2_cls_ssg.get_model(num_class, normal_channel=False)
classifier = classifier.cuda()

checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model_epoch_180.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
    
pred_choices = []
probs = []
prob_collision_list = []
prob_no_collision_list = []
point_clouds = pickle.load(open('real_world_data/pcd_0001.pickle', 'rb'))

visualize_pointcloud_and_action_pt_selection(point_clouds[0])

np.random.seed(0)
for i in range(50):
    actions = np.zeros((1, 10))
    actions[:, -1] = -1.0
    # x = np.random.uniform(-0.02, 0.02)
    x = np.random.uniform(0.05, 0.1)
    y = np.random.uniform(-0.02, 0.02)
    # z = np.random.uniform(-0.1, 0.2)
    z = np.random.uniform(-0.22, -0.18)
    actions[:, 3:6] = np.array([x, y, z]) 

    pred_choice, prob = predict_collisions(points=point_clouds, actions=actions)
    pred_choices.append(pred_choice)
    probs.append(prob)

    if pred_choice == 1.0:
        prob_collision_list.append(prob)
    else:
        prob_no_collision_list.append(prob)

print("len(prob_collision_list): ", len(prob_collision_list))
print("len(prob_no_collision_list): ", len(prob_no_collision_list))

fig, axs = plt.subplots(1, 2)
axs[0].hist(prob_collision_list, bins=20, alpha=0.5, label='Collision')
axs[1].hist(prob_no_collision_list, bins=20, alpha=0.5, label='No Collision')
# axs[0].legend(loc='upper right')
axs[0].set_title('Collisions (should be high)')
axs[1].set_title('No Collisions (should be low)')
axs[0].set_xlabel('Probability')
axs[1].set_xlabel('Probability')
axs[0].set_ylim(0, 30)
axs[1].set_ylim(0, 30)
axs[0].set_xlim(0, 1.0)
axs[1].set_xlim(0, 1.0)
axs[0].set_ylabel('Frequency')
axs[1].set_ylabel('Frequency')
plt.show()

# visualize_pointcloud_and_action(point_clouds[waypoint_number],
#                                 point_colors[waypoint_number],
#                                 point_clouds[waypoint_number + 1],
#                                 point_colors[waypoint_number + 1],
#                                 action=actions[waypoint_number],
#                                 eef_pos=eef_pos[waypoint_number])