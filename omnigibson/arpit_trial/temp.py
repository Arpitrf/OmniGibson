from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
import pickle
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R

# f_name = "0001.pickle"
# with open(f"/home/arpit/test_projects/OmniGibson/real_world_data/{f_name}", "rb") as f:
#     data_dict = pickle.load(f)
#     for k in data_dict.keys():
#         print("k, v: ", k, np.array(data_dict[k]).shape)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.array(data_dict["rgb"]))
# ax[1].imshow(np.array(data_dict["depth"]))
# plt.show()

# cv2.imwrite(f"/home/arpit/test_projects/OmniGibson/real_world_data/{f_name}_rgb.png", np.array(data_dict["rgb"]))
# cv2.imwrite(f"/home/arpit/test_projects/OmniGibson/real_world_data/{f_name}_depth.png", np.array(data_dict["depth"]))
# breakpoint()


# import matplotlib.pyplot as plt
# import numpy as np
# from collections import Counter
# # Create a dictionary to store phase-reason pairs
# phase_reason_pairs = {}
# for i, phase in enumerate(phases):
#     reason = reasons[i]
#     if phase not in phase_reason_pairs:
#         phase_reason_pairs[phase] = Counter()
#     phase_reason_pairs[phase][reason] += 1

# # Set up the plot
# fig, ax = plt.subplots(figsize=(12, 6))

# # Get unique reasons for the legend
# unique_reasons = set(reasons)
# colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_reasons)))
# reason_color_map = dict(zip(unique_reasons, colors))

# # Plot stacked bars for each phase
# x = np.arange(len(phase_reason_pairs))
# bottom = np.zeros(len(phase_reason_pairs))

# for reason in unique_reasons:
#     heights = [phase_reason_pairs[phase][reason] for phase in phase_reason_pairs.keys()]
#     ax.bar(x, heights, bottom=bottom, label=reason, color=reason_color_map[reason])
    
#     # Add value labels for non-zero counts
#     for i, height in enumerate(heights):
#         if height > 0:
#             ax.text(x[i], bottom[i] + height/2, str(int(height)), 
#                    ha='center', va='center')
#     bottom += heights

# # Customize the plot
# ax.set_ylabel('Count')
# ax.set_title('Distribution of Errors by Phase and Reason')
# ax.set_xticks(x)
# ax.set_xticklabels(list(phase_reason_pairs.keys()), rotation=45, ha='right')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.show()

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

def get_seg_instance_id_info(ep, hdf5_file):
    # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id_id
    if 'seg_instance_id_strings' in hdf5_file[f'data/{ep}/observations_info'].keys():
        seg_instance_id_strings = np.array(hdf5_file["data/{}/observations_info/seg_instance_id_strings".format(ep)])
        seg_instance_id_shapes = np.array(hdf5_file["data/{}/observations_info/seg_instance_id_shapes".format(ep)])
        seg_instance_id = extract_observations_info_from_hdf5(obs_info_strings=seg_instance_id_strings, 
                                                                    obs_info_shapes=seg_instance_id_shapes)
        # print("111: ", seg_instance_id.shape)
    else:
        hd5key = "data/{}/observations_info/seg_instance_id".format(ep)
        # seg_instance_id = hdf5_file[hd5key]
        seg_instance_id = np.array(hdf5_file[hd5key]).astype(str)
        # print("222: ", seg_instance_id.shape)
    return seg_instance_id

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


def show_vectors(hdf5_file):
     # show the original vector and the noisy vector in matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    # ax.quiver(0, 0, 0, original_vector[0], original_vector[1], original_vector[2], color='r')
    for i, ep in enumerate(hdf5_file["data"]):
        # if i != 7:
        #     continue
        vector = np.array(hdf5_file[f"data/{ep}/actions/complete_actions"])[0]
        grasp_vector = np.array(hdf5_file[f"data/{ep}/extras/grasps"])
        print("Episode: ", ep)
        print("vector: ", vector)
        print("grasp_vector: ", grasp_vector)
        print("==============")
        color = "g"
        if any(grasp_vector == False):
            color = "r"
        ax.text(vector[0], vector[1], vector[2], f"{i}", color=color)
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color)
        # breakpoint()
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.2, 0.2])
    plt.show()

def visualize_trajectories(actions, start_position, ax=None, color='r', ep=0, grasp_vector=None, base_orn=None): 
    total_lines = 0
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        prev_position = start_position

        for j in range(trajectory.shape[0]):
            direction = trajectory[j]  # Direction vector at this waypoint
            # convert delta from robot frame to world frame
            if base_orn is not None:
                base_orn_matrix = R.from_quat(base_orn).as_matrix()
                direction = base_orn_matrix @ direction
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction
            step = magnitude * direction_normalized

            next_position = prev_position + step
            if grasp_vector is not None:
                if grasp_vector[j+1]:
                    color = "g"
                else:
                    color = "r"
            ax.quiver(prev_position[0], prev_position[1], prev_position[2], step[0], step[1], step[2], color=color)
            # ax.quiver(0, 0, 0, step[0], step[1], step[2], color=color)
            # visualize_marker(start_position=prev_position, end_position=next_position, id=total_lines)
            prev_position = prev_position + step  # Move to the new position
            total_lines += 1

        # ax.text(next_position[0], next_position[1], next_position[2], f"{ep}", color=color)

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

# # visualize a pcd
# import open3d as o3d
# path = "/home/arpit/projects/OmniGibson/pcd_1.ply"
# pcd = o3d.io.read_point_cloud(path)
# o3d.visualization.draw_geometries([pcd])






import h5py
import numpy as np
with h5py.File("/home/arpit/projects/OmniGibson/place_in_shelf/place_in_shelf_robomimic_dataset.hdf5", "r") as f:
    print(len(f["data"].keys()))
    breakpoint()
    
    # # Inspect the keys
    count = np.zeros(8)
    for i in range(len(f["data"])):
    #     # checking number of waypoints in each episode
    #     # label = np.array(f[f"data/episode_{i:05d}/extras/grasps"])
        # actions = np.array(f[f"data/episode_{i:05d}/actions/actions"])
        actions = np.array(f[f"data/demo_{i}/actions"])
    #     label = np.array(f[f"data/episode_{i:05d}/extras/base_collision"])
    #     scan = np.array(f[f"data/episode_{i:05d}/observations/scan"])
        # print("actions.shape: ", actions.shape)
        count[actions.shape[0]] += 1

    print("count: ", count)

    # # ======== Check number of episodes with no collision and all waypoints ==========
    # succ_ep = 0
    # for i in range(len(f["data"])):
    #     contact = False
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/contacts"])):
    #         if f[f"data/episode_{i:05d}/extras/contacts"][j]:
    #             contact = True
    #             break
    #     if not contact and actions[0] == 4: # 6 for place_in_shelf and 4 for place_in_drawer
    #         succ_ep += 1
    #         print(i)
    # print("succ_ep: ", succ_ep)
    # breakpoint()
    # # ============================================================


    #     print("scan, action, label shapes: ", scan.shape, actions.shape, label.shape)
    
    # print("lis: ", lis)
    # print("lis2: ", lis2)
    
        # actions = np.array(f[f"data/episode_{i:05d}/actions/actions"])
        # print("actions: ", actions)
    #     obs = np.array(f[f"data/episode_{i:05d}/observations/rgb"])
    #     supervision = np.array(f[f"data/episode_{i:05d}/extras/contacts"])
    #     print("actions: ", actions.shape)
    #     print("obs: ", obs.shape)
    #     print("supervision: ", supervision)
    #     print("actions: ", actions)
    #     print("supervision: ", supervision)
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(obs[0, :, :, :3])
    #     ax[1].imshow(obs[1, :, :, :3])
    #     plt.show()
    
    # grasp_label = np.array(f["data/episode_00000/extras/grasps"])
    # print("grasp_label: ", grasp_label)
    # breakpoint()


    # ep = "episode_00005"
    # # print observations info
    # seg_semantic_info = get_seg_semantic_info(f"{ep}", f)
    # seg_instance_info = get_seg_instance_info(f"{ep}", f)
    # seg_instance_id_info = get_seg_instance_id_info(f"{ep}", f)
    # # print("seg_semantic_info: ", seg_semantic_info)
    # # print("---------------")
    # # print("seg_instance_info: ", seg_instance_info)
    # print("---------------")
    # print("seg_instance_id_info: ", seg_instance_id_info)
    # # Show the observations
    # fig, ax = plt.subplots(2,2)
    # seg_semantic = f[f'data/{ep}/observations/seg_semantic']
    # seg_instance = f[f'data/{ep}/observations/seg_instance']
    # seg_instance_id = f[f'data/{ep}/observations/seg_instance_id']
    # rgb = f[f'data/{ep}/observations/rgb']
    # ax[0, 0].imshow(rgb[-1, :, :, :3])
    # ax[0, 1].imshow(seg_semantic[-1])
    # ax[1, 0].imshow(seg_instance[-1])
    # ax[1, 1].imshow(seg_instance_id[-1])
    # plt.show()
    
    
    # show_vectors(f)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(f["data"])):
    #     if i > 10:
    #         break
    #     print("Episode: ", i)
    #     print("len(actions), len(obs): ", np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape, np.array(f[f"data/episode_{i:05d}/observations/rgb"]).shape)
    #     print("grasps: ", np.array(f[f"data/episode_{i:05d}/extras/grasps"]))

    #     if len(np.array(f[f"data/episode_{i:05d}/actions/actions"])) == 0:
    #         continue
    #     # print("actions: ",  np.array(f[f"data/episode_{i:05d}/actions/actions"])[:, 3:6])
    #     action_traj = np.array(f[f"data/episode_{i:05d}/actions/actions"])
    #     grasp_vector = np.array(f[f"data/episode_{i:05d}/extras/grasp_label"])
    #     base_orn = np.array(f[f"data/episode_{i:05d}/proprioceptions/base_orn"])[0]
    #     visualize_trajectories(action_traj[None, ...], np.array([0.0, 0.0, 0.0]), ax=ax, ep=i, grasp_vector=grasp_vector, base_orn=base_orn)
    # plt.show()

    # # ================= Inspect grasp supervision =================
    # total_actions = 0
    # yes = 0
    # no = 0
    # for i in range(len(f["data"])):
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     total_actions += actions[0]
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/grasps"])):
    #         if not f[f"data/episode_{i:05d}/extras/grasps"][j] and not f[f"data/episode_{i:05d}/extras/ft_label"][j]:
    #             yes += 1
    #         else:
    #             no += 1
    #             # fig, ax = plt.subplots(2)
    #             # rgb_1 = f[f'data/episode_{i:05d}/observations/rgb'][j-1, :, :, :3]
    #             # rgb_2 = f[f'data/episode_{i:05d}/observations/rgb'][j, :, :, :3]
    #             # ax[0].imshow(rgb_1)
    #             # ax[1].imshow(rgb_2)
    #             # plt.show()
    #     #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    # print("total_actions: ", total_actions)
    # print("grasps: ", yes)
    # print("no_grasps: ", no)
    # # ============================================================

    # # ================= Inspect grasp supervision =================
    # total_actions = 0
    # grasps = 0
    # no_grasps = 0
    # for i in range(len(f["data"])):
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     total_actions += actions[0]
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/grasp_label"])):
    #         if f[f"data/episode_{i:05d}/extras/grasp_label"][j]:
    #             grasps += 1
    #         else:
    #             no_grasps += 1
    #             # fig, ax = plt.subplots(2)
    #             # rgb_1 = f[f'data/episode_{i:05d}/observations/rgb'][j-1, :, :, :3]
    #             # rgb_2 = f[f'data/episode_{i:05d}/observations/rgb'][j, :, :, :3]
    #             # ax[0].imshow(rgb_1)
    #             # ax[1].imshow(rgb_2)
    #             # plt.show()
    #     #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    # print("total_actions: ", total_actions)
    # print("grasps: ", grasps)
    # print("no_grasps: ", no_grasps)
    # # ============================================================

    # # ================= Inspect F/T supervision =================
    # total_actions = 0
    # ft = 0
    # no_ft = 0
    # for i in range(len(f["data"])):
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     total_actions += actions[0]
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/ft_label"])):
    #         if f[f"data/episode_{i:05d}/extras/ft_label"][j]:
    #             ft += 1
    #         else:
    #             no_ft += 1
    #             # fig, ax = plt.subplots(2)
    #             # rgb_1 = f[f'data/episode_{i:05d}/observations/rgb'][j-1, :, :, :3]
    #             # rgb_2 = f[f'data/episode_{i:05d}/observations/rgb'][j, :, :, :3]
    #             # ax[0].imshow(rgb_1)
    #             # ax[1].imshow(rgb_2)
    #             # plt.show()
    #     #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    # print("total_actions: ", total_actions)
    # print("ft: ", ft)
    # print("no_ft: ", no_ft)
    # # ============================================================

    # ================= Inspect collision supervision =================
    total_actions = 0
    contacts = 0
    no_contacts = 0
    for i in range(len(f["data"])):
        actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
        total_actions += actions[0]
        for j in range(1, len(f[f"data/episode_{i:05d}/extras/contacts"])):
            if f[f"data/episode_{i:05d}/extras/contacts"][j]:
                contacts += 1
            else:
                no_contacts += 1
        #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    print("total_actions: ", total_actions)
    print("contacts: ", contacts)
    print("no_contacts: ", no_contacts)
    # ============================================================

    # ================= Inspect object dropping supervision =================
    total_actions = 0
    object_dropped = 0
    no_object_dropped = 0
    for i in range(len(f["data"])):
        actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
        total_actions += actions[0]
        for j in range(1, len(f[f"data/episode_{i:05d}/extras/object_dropped"])):
            if f[f"data/episode_{i:05d}/extras/object_dropped"][j]:
                object_dropped += 1
            else:
                no_object_dropped += 1
        #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    print("total_actions: ", total_actions)
    print("object_dropped: ", object_dropped)
    print("no_object_dropped: ", no_object_dropped)
    # ============================================================

    # # ================= Inspect singularities/joint limits supervision =================
    # total_actions = 0
    # trues = 0
    # falses = 0
    # for i in range(len(f["data"])):
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     total_actions += actions[0]
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/joint_limits"])):
    #         if f[f"data/episode_{i:05d}/extras/joint_limits"][j]:
    #             trues += 1
    #         else:
    #             falses += 1
    #     #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    # print("total_actions: ", total_actions)
    # print("trues: ", trues)
    # print("falses: ", falses)
    # # ============================================================   
    # 
    # # # ================= Inspect base collision supervision =================
    # total_actions = 0
    # contacts = 0
    # no_contacts = 0
    # for i in range(len(f["data"])):
    #     actions = np.array(f[f"data/episode_{i:05d}/actions/actions"]).shape
    #     total_actions += actions[0]
    #     for j in range(1, len(f[f"data/episode_{i:05d}/extras/base_collision"])):
    #         if f[f"data/episode_{i:05d}/extras/base_collision"][j]:
    #             contacts += 1
    #         else:
    #             no_contacts += 1
    #     #     action = np.array(f[f"data/episode_{i:05d}/actions/actions"])[j]
    # print("total_actions: ", total_actions)
    # print("contacts: ", contacts)
    # print("no_contacts: ", no_contacts)
    # # ============================================================ 
    
    # ========================== Check if ground plane exists ==========================
    # no_ground = 0
    # for i in range(len(f["data"])):
    #     # print(f["data"]["episode_{:05d}".format(i)]["observations_info"].keys())
    #     seg_instance_info = get_seg_instance_info(f"episode_{i:05d}", f)
    #     # print(np.array(seg_instance_info).shape)
        
    #     for waypt in seg_instance_info:
    #         ground_exist = False
    #         for j in range(len(waypt)):
    #             if "groundPlane" == waypt[j][1]:
    #                 ground_exist = True
    #                 break
    #         if not ground_exist:
    #             # breakpoint()
    #             fig, ax = plt.subplots(2,2)
    #             seg_semantic = f[f'data/episode_{i:05d}/observations/seg_semantic']
    #             seg_instance = f[f'data/episode_{i:05d}/observations/seg_instance']
    #             seg_instance_id = f[f'data/episode_{i:05d}/observations/seg_instance_id']
    #             rgb = f[f'data/episode_{i:05d}/observations/rgb']
    #             ax[0, 0].imshow(rgb[-1, :, :, :3])
    #             ax[0, 1].imshow(seg_semantic[-1])
    #             ax[1, 0].imshow(seg_instance[-1])
    #             ax[1, 1].imshow(seg_instance_id[-1])
    #             plt.show()
    #             print(f"Episode {i} has no ground plane")
    #             # camera_pos = np.array(f[f'data/episode_{i:05d}/proprioceptions/camera_qpos'])[0][1]
    #             # print(camera_pos)
    #             no_ground += 1
    #             break

    # print("no_ground: ", no_ground)
    # ===================================================================================

    # # ========================== Check if key exists ==========================
    # no_ground = 0
    # # for i in range(len(f["data"])):
    # for i in range(1600):
    #     # print(f["data"]["episode_{:05d}".format(i)]["observations_info"].keys())
    #     seg_instance_info = get_seg_instance_info(f"episode_{i:05d}", f)
    #     seg_instance_id_info = get_seg_instance_id_info(f"episode_{i:05d}", f)
        
    #     for waypt in seg_instance_id_info:
    #         ground_exist = False
    #         for j in range(len(waypt)):
    #             # if "groundPlane" == waypt[j][1]:
    #             if "bottom_cabinet/link_7" in waypt[j][1]:
    #                 ground_exist = True
    #                 break
    #         if not ground_exist:
    #             # breakpoint()
    #             fig, ax = plt.subplots(2,2)
    #             seg_semantic = f[f'data/episode_{i:05d}/observations/seg_semantic']
    #             seg_instance = f[f'data/episode_{i:05d}/observations/seg_instance']
    #             seg_instance_id = f[f'data/episode_{i:05d}/observations/seg_instance_id']
    #             rgb = f[f'data/episode_{i:05d}/observations/rgb']
    #             ax[0, 0].imshow(rgb[-1, :, :, :3])
    #             ax[0, 1].imshow(seg_semantic[-1])
    #             ax[1, 0].imshow(seg_instance[-1])
    #             ax[1, 1].imshow(seg_instance_id[-1])
    #             plt.show()
    #             print(f"Episode {i} has no ground plane")
    #             # camera_pos = np.array(f[f'data/episode_{i:05d}/proprioceptions/camera_qpos'])[0][1]
    #             # print(camera_pos)
    #             no_ground += 1
    #             break

    # print("no_ground: ", no_ground)
    # # ===================================================================================

    #     # if "seg_instance_strings" in f["data"]["episode_{:05d}".format(i)]["observations_info"].keys():
    #     #     print(f"Episode {i} has seg_instance_strings")
    #     actions_shape = np.array(f["data"]["episode_{:05d}".format(i)]["actions"]["actions"]).shape
    #     rgb_shape = np.array(f["data"]["episode_{:05d}".format(i)]["observations"]["rgb"]).shape
    #     contacts_shape = np.array(f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"]).shape
    #     if actions_shape[0] != rgb_shape[0] - 1 and actions_shape[0] != contacts_shape[0] - 1:
    #         print(f"Episode {i} has incorrect shape")
    #     # print("actions: ", np.array(f["data"]["episode_{:05d}".format(i)]["actions"]["actions"]).shape)
    #     # print("rgb: ", np.array(f["data"]["episode_{:05d}".format(i)]["observations"]["rgb"]).shape)
    #     # print("contacts: ", np.array(f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"]).shape)
    #     # print("singularities: ", np.array(f["data"]["episode_{:05d}".format(i)]["extras"]["singularities"]).shape)
    #     # print("extrinsic_matrix: ", np.array(f["data"]["episode_{:05d}".format(i)]["proprioceptions"]["extrinsic_matrix"]).shape)
    #     # print(("--------------"))

#     print(np.array(f["data"]["episode_00250"]["actions"]["actions"]))
#     print(np.array(f["data"]["episode_00250"]["proprioceptions"]["right_eef_pos"]).shape)
#     print(np.array(f["data"]["episode_00250"]["extras"]["contacts"]))











# def filter_hdf5_episodes(input_path, output_path, episode_keys):
#     """
#     Read a hdf5 file and create a new hdf5 file containing only specified episodes.
    
#     Args:
#         input_path (str): Path to input hdf5 file
#         output_path (str): Path where filtered hdf5 file will be saved
#         episode_keys (list): List of episode keys to keep in filtered file
#     """
#     # Open input file in read mode
#     with h5py.File(input_path, 'r') as src:
#         # Create output file
#         with h5py.File(output_path, 'w') as dst:
#             # Copy over all attributes from source file
#             for key, val in src.attrs.items():
#                 dst.attrs[key] = val
            
#             # Copy data group
#             if 'data' in src:
#                 data_group = dst.create_group('data')
#                 # Only copy specified episodes
#                 for ep in episode_keys:
#                     if ep in src['data']:
#                         src.copy(f'/data/{ep}', data_group)
            
#             # Copy mask group if it exists
#             if 'mask' in src:
#                 mask_group = dst.create_group('mask')
#                 # Copy all mask datasets
#                 for key in src['mask'].keys():
#                     # Get original mask data
#                     orig_mask = src['mask'][key][:]
#                     # Filter to only include specified episodes
#                     filtered_mask = [x for x in orig_mask if x.decode('utf-8') in episode_keys]
#                     # Create new dataset with filtered mask
#                     mask_group.create_dataset(key, data=filtered_mask)

#     return

# filter_hdf5_episodes("/home/arpit/test_projects/OmniGibson/place_in_shelf_data/dataset.hdf5", "/home/arpit/test_projects/OmniGibson/place_in_shelf_data/filtered_dataset.hdf5", ["episode_00250"])



# def overlay_images():
#     import cv2
#     import numpy as np
#     import os
#     import matplotlib.pyplot as plt

#     # Path to your folder with video files
#     folder_path = "/home/arpit/projects/OmniGibson/open_drawer_temp3"

#     # Get a sorted list of video files
#     video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])

#     # List to store the frames
#     frames = []

#     for video_file in video_files:
#         video_path = os.path.join(folder_path, video_file)
#         cap = cv2.VideoCapture(video_path)
#         ret, frame = cap.read()  # Read the first frame
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
#             frames.append(frame)
#         cap.release()

#     # Ensure we have frames to process
#     if frames:
#         # Convert the list of frames to a numpy array and calculate the mean image
#         # overlay_image = np.mean(frames, axis=0).astype(np.uint8)
#         overlay_image = np.zeros_like(frames[0], dtype=np.float32)
#         for frame in frames:
#             overlay_image += frame.astype(np.float32) / len(frames)
#         overlay_image = overlay_image.astype(np.uint8)

#         # Display the result using matplotlib
#         plt.imshow(overlay_image)
#         plt.axis('off')
#         plt.title('Overlay of First Frames')
#         plt.show()
#     else:
#         print("No frames extracted. Check your folder path or video files.")

# overlay_images()